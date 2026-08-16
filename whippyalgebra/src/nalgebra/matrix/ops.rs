#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use whippyunits::quantity::{Quantity, Unit};

use super::MixedUnitMatrix;
use crate::dims::{
    CrossMul, CrossMulled, DiagUnit, DivBy, MapUnits, Mapped, MulBy, Reciprocal, RescaleFactors,
    UniformDiag, ZipDiv, ZipDivided, ZipMul, ZipMulled,
};
use crate::entry::FromRaw;

// Component-wise (Hadamard) product and quotient. Unlike matrix `Mul` (which
// contracts an inner dimension) or `Add`/`Sub` (which require identical unit
// grids), these impose *no* dimensional coherence beyond a matching shape and
// brand: the representable unit-matrices form a linear subspace in log-unit
// space, and Hadamard mul/div are just addition/subtraction within it. The
// result's row and column dimension lists are the element-wise product /
// quotient of the operands' respective lists ([`ZipMul`] / [`ZipDiv`]).
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Component-wise (Hadamard) product with another matrix of the same shape
    /// and brand.
    ///
    /// Entry `(i, j)` of the result is `self(i,j) · rhs(i,j)`, whose unit is
    /// `(RowDims[i] · RowDimsB[i]) / (ColDims[j] · ColDimsB[j])`; the result's
    /// dimension lists are thus the element-wise products of the operands'.
    pub fn component_mul<RowDimsB, ColDimsB, SB>(
        &self,
        rhs: &MixedUnitMatrix<RowDimsB, ColDimsB, nalgebra::Matrix<T, R, C, SB>, Brand>,
    ) -> MixedUnitMatrix<
        ZipMulled<RowDims, RowDimsB>,
        ZipMulled<ColDims, ColDimsB>,
        nalgebra::OMatrix<T, R, C>,
        Brand,
    >
    where
        T: Mul<T, Output = T>,
        SB: nalgebra::RawStorage<T, R, C>,
        RowDims: ZipMul<RowDimsB>,
        ColDims: ZipMul<ColDimsB>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        MixedUnitMatrix::from_nalgebra(self.inner.zip_map(&rhs.inner, |a, b| a * b))
    }

    /// Component-wise (Hadamard) quotient with another matrix of the same shape
    /// and brand.
    ///
    /// Entry `(i, j)` of the result is `self(i,j) / rhs(i,j)`, whose unit is
    /// `(RowDims[i] / RowDimsB[i]) / (ColDims[j] / ColDimsB[j])`; the result's
    /// dimension lists are thus the element-wise quotients of the operands'.
    pub fn component_div<RowDimsB, ColDimsB, SB>(
        &self,
        rhs: &MixedUnitMatrix<RowDimsB, ColDimsB, nalgebra::Matrix<T, R, C, SB>, Brand>,
    ) -> MixedUnitMatrix<
        ZipDivided<RowDims, RowDimsB>,
        ZipDivided<ColDims, ColDimsB>,
        nalgebra::OMatrix<T, R, C>,
        Brand,
    >
    where
        T: Div<T, Output = T>,
        SB: nalgebra::RawStorage<T, R, C>,
        RowDims: ZipDiv<RowDimsB>,
        ColDims: ZipDiv<ColDimsB>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        MixedUnitMatrix::from_nalgebra(self.inner.zip_map(&rhs.inner, |a, b| a / b))
    }
}

// Kronecker (tensor) product. Like the component-wise ops, it imposes no
// dimensional coherence beyond a matching brand; the shapes multiply. The
// result's row index `i1 * rows2 + i2` carries the unit `RowDims[i1] ·
// RowDimsB[i2]` and its column index `j1 * cols2 + j2` the unit `ColDims[j1] ·
// ColDimsB[j2]`, so both dimension lists are the outer product ([`CrossMul`])
// of the operands' — and entry `((i1,i2), (j1,j2))` works out to the product of
// the two operands' entry units, exactly as a tensor product of linear maps.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// The Kronecker (tensor) product with another matrix of the same brand.
    ///
    /// For a `p x q` `self` and an `r x s` `rhs`, the result is `pr x qs`, with
    /// row dimension list `CrossMul(RowDims, RowDimsB)` and column dimension
    /// list `CrossMul(ColDims, ColDimsB)`. Each result entry is the product of
    /// the corresponding `self` and `rhs` entries, so its unit is likewise the
    /// product of their entry units.
    pub fn kronecker<RowDimsB, ColDimsB, R2, C2, SB>(
        &self,
        rhs: &MixedUnitMatrix<RowDimsB, ColDimsB, nalgebra::Matrix<T, R2, C2, SB>, Brand>,
    ) -> MixedUnitMatrix<
        CrossMulled<RowDims, RowDimsB>,
        CrossMulled<ColDims, ColDimsB>,
        nalgebra::OMatrix<T, nalgebra::DimProd<R, R2>, nalgebra::DimProd<C, C2>>,
        Brand,
    >
    where
        T: nalgebra::ComplexField,
        R2: nalgebra::Dim,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, R2, C2>,
        R: nalgebra::DimMul<R2>,
        C: nalgebra::DimMul<C2>,
        RowDims: CrossMul<RowDimsB>,
        ColDims: CrossMul<ColDimsB>,
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::DimProd<R, R2>, nalgebra::DimProd<C, C2>>,
    {
        MixedUnitMatrix::from_nalgebra(self.inner.kronecker(&rhs.inner))
    }
}

impl<RowA, Inner, ColB, Brand, MA, MB> Mul<MixedUnitMatrix<Inner, ColB, MB, Brand>>
    for MixedUnitMatrix<RowA, Inner, MA, Brand>
where
    MA: Mul<MB>,
{
    type Output = MixedUnitMatrix<RowA, ColB, <MA as Mul<MB>>::Output, Brand>;

    fn mul(self, rhs: MixedUnitMatrix<Inner, ColB, MB, Brand>) -> Self::Output {
        MixedUnitMatrix {
            inner: self.inner * rhs.inner,
            _dims: PhantomData,
        }
    }
}

impl<RowDims, ColDims, Brand, MA, MB> Add<MixedUnitMatrix<RowDims, ColDims, MB, Brand>>
    for MixedUnitMatrix<RowDims, ColDims, MA, Brand>
where
    MA: Add<MB>,
{
    type Output = MixedUnitMatrix<RowDims, ColDims, <MA as Add<MB>>::Output, Brand>;

    fn add(self, rhs: MixedUnitMatrix<RowDims, ColDims, MB, Brand>) -> Self::Output {
        MixedUnitMatrix {
            inner: self.inner + rhs.inner,
            _dims: PhantomData,
        }
    }
}

impl<RowDims, ColDims, Brand, MA, MB> Sub<MixedUnitMatrix<RowDims, ColDims, MB, Brand>>
    for MixedUnitMatrix<RowDims, ColDims, MA, Brand>
where
    MA: Sub<MB>,
{
    type Output = MixedUnitMatrix<RowDims, ColDims, <MA as Sub<MB>>::Output, Brand>;

    fn sub(self, rhs: MixedUnitMatrix<RowDims, ColDims, MB, Brand>) -> Self::Output {
        MixedUnitMatrix {
            inner: self.inner - rhs.inner,
            _dims: PhantomData,
        }
    }
}

impl<RowDims, ColDims, Brand, MA, MB> AddAssign<MixedUnitMatrix<RowDims, ColDims, MB, Brand>>
    for MixedUnitMatrix<RowDims, ColDims, MA, Brand>
where
    MA: AddAssign<MB>,
{
    fn add_assign(&mut self, rhs: MixedUnitMatrix<RowDims, ColDims, MB, Brand>) {
        self.inner += rhs.inner;
    }
}

impl<RowDims, ColDims, Brand, MA, MB> SubAssign<MixedUnitMatrix<RowDims, ColDims, MB, Brand>>
    for MixedUnitMatrix<RowDims, ColDims, MA, Brand>
where
    MA: SubAssign<MB>,
{
    fn sub_assign(&mut self, rhs: MixedUnitMatrix<RowDims, ColDims, MB, Brand>) {
        self.inner -= rhs.inner;
    }
}

impl<RowDims, ColDims, Brand, M> Neg for MixedUnitMatrix<RowDims, ColDims, M, Brand>
where
    M: Neg,
{
    type Output = MixedUnitMatrix<RowDims, ColDims, <M as Neg>::Output, Brand>;

    fn neg(self) -> Self::Output {
        MixedUnitMatrix {
            inner: -self.inner,
            _dims: PhantomData,
        }
    }
}

// Scalar-by-quantity multiplication scales every entry's unit by the scalar
// quantity's unit. Adopting the convention that the scalar acts on the *row*
// (output) space, this multiplies each `RowDims` entry by the scalar's unit and
// leaves `ColDims` untouched; entry `(i, j)` goes from `row_i / col_j` to
// `(q · row_i) / col_j = q · (row_i / col_j)`, as required. The brand must match.
impl<RowDims, ColDims, Brand, T, R, C, S, Sq, Dq> Mul<Quantity<Unit<Sq, Dq>, T, Brand>>
    for MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    nalgebra::Matrix<T, R, C, S>: Mul<T>,
    RowDims: MapUnits<MulBy<Unit<Sq, Dq>>>,
{
    type Output = MixedUnitMatrix<
        Mapped<MulBy<Unit<Sq, Dq>>, RowDims>,
        ColDims,
        <nalgebra::Matrix<T, R, C, S> as Mul<T>>::Output,
        Brand,
    >;

    fn mul(self, rhs: Quantity<Unit<Sq, Dq>, T, Brand>) -> Self::Output {
        MixedUnitMatrix {
            inner: self.inner * rhs.unsafe_value,
            _dims: PhantomData,
        }
    }
}

// Scalar-by-quantity division divides every row entry's unit by the scalar's.
impl<RowDims, ColDims, Brand, T, R, C, S, Sq, Dq> Div<Quantity<Unit<Sq, Dq>, T, Brand>>
    for MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    nalgebra::Matrix<T, R, C, S>: Div<T>,
    RowDims: MapUnits<DivBy<Unit<Sq, Dq>>>,
{
    type Output = MixedUnitMatrix<
        Mapped<DivBy<Unit<Sq, Dq>>, RowDims>,
        ColDims,
        <nalgebra::Matrix<T, R, C, S> as Div<T>>::Output,
        Brand,
    >;

    fn div(self, rhs: Quantity<Unit<Sq, Dq>, T, Brand>) -> Self::Output {
        MixedUnitMatrix {
            inner: self.inner / rhs.unsafe_value,
            _dims: PhantomData,
        }
    }
}

// Scaling by a bare (dimensionless) real leaves every entry's unit unchanged.
//
// We expose this as `scale`/`unscale` methods rather than the `*`/`/` operators:
// a blanket `Mul<T>` for the raw scalar would collide with the matrix-by-matrix
// `Mul<MixedUnitMatrix<…>>` impl, because the scalar type parameter can unify
// with the right-hand matrix type. (Multiplying by a dimensionless `Quantity`
// via the operator remains available and is equivalent.)
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::SimdComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// Multiplies every entry by a real scalar, leaving all units unchanged.
    pub fn scale(
        &self,
        factor: T::SimdRealField,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        MixedUnitMatrix::from_nalgebra(self.inner.scale(factor))
    }

    /// Divides every entry by a real scalar, leaving all units unchanged.
    pub fn unscale(
        &self,
        factor: T::SimdRealField,
    ) -> MixedUnitMatrix<RowDims, ColDims, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        MixedUnitMatrix::from_nalgebra(self.inner.unscale(factor))
    }
}

impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::SimdComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::StorageMut<T, R, C>,
{
    /// Multiplies every entry by a real scalar in place, leaving units unchanged.
    pub fn scale_mut(&mut self, factor: T::SimdRealField) {
        self.inner.scale_mut(factor);
    }

    /// Divides every entry by a real scalar in place, leaving units unchanged.
    pub fn unscale_mut(&mut self, factor: T::SimdRealField) {
        self.inner.unscale_mut(factor);
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
    /// Returns the transpose.
    ///
    /// Transpose is not a plain dimension swap: `Mᵀ(i, j) = M(j, i)` has unit
    /// `row_j / col_i`, so the transposed row/column dimension lists are the
    /// reciprocals of the original column/row lists (contrast
    /// [`try_inverse`](MixedUnitMatrix::try_inverse), which is a clean swap).
    pub fn transpose(
        &self,
    ) -> MixedUnitMatrix<
        Mapped<Reciprocal, ColDims>,
        Mapped<Reciprocal, RowDims>,
        nalgebra::OMatrix<T, C, R>,
        Brand,
    >
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, R>,
        ColDims: MapUnits<Reciprocal>,
        RowDims: MapUnits<Reciprocal>,
    {
        MixedUnitMatrix {
            inner: self.inner.transpose(),
            _dims: PhantomData,
        }
    }
}

/// Rescales a matrix to different units of the same dimensions — the matrix
/// analog of whippyunits' [`rescale`](whippyunits::api::rescale()), acting on
/// every row and every column at once.
///
/// Where `rescale` reexpresses a single quantity in a new scale of its
/// dimension, `rescale_matrix` reexpresses each row unit and each column unit in
/// a new scale of the same dimension at once: the source `⟨RowDims, ColDims⟩`
/// becomes `⟨NewRowDims, NewColDims⟩`, where each new unit has the same
/// dimension as the old (only the scale differs). Entry `(i, j)` is converted
/// from `RowDims[i] / ColDims[j]` to `NewRowDims[i] / NewColDims[j]`, its value
/// multiplied by `rowfactor[i] / colfactor[j]`. A target list whose length or
/// dimensions do not match the source has no [`RescaleFactors`] impl and is a
/// compile error.
///
/// The target dimension lists are read from the return type, exactly like
/// `rescale` reads its target unit from the annotated result:
///
/// ```
/// use whippyalgebra::dims;
/// use whippyalgebra::nalgebra::{mixed_unit_matrix, rescale_matrix, MixedUnitMatrix, OMatrix, Const};
/// use whippyunits::{quantity, qty};
///
/// type Rows = dims![m, m / s];
/// type Cols = dims![s];
/// let m = mixed_unit_matrix![Rows, Cols;
///     [quantity!(1.0, m / s)],
///     [quantity!(2.0, m / s^2)],
/// ];
///
/// // Rescale the rows to mm and mm/s, keeping the column in s.
/// type NewRows = dims![mm, mm / s];
/// let mm: MixedUnitMatrix<NewRows, Cols, OMatrix<f64, Const<2>, Const<1>>> =
///     rescale_matrix(&m);
///
/// let a: qty!(mm / s) = mm.get::<0, 0>();
/// let b: qty!(mm / s^2) = mm.get::<1, 0>();
/// assert!((a.unsafe_value - 1000.0).abs() < 1e-9);
/// assert!((b.unsafe_value - 2000.0).abs() < 1e-9);
/// ```
pub fn rescale_matrix<RowDims, ColDims, NewRowDims, NewColDims, Brand, T, R, C, S>(
    matrix: &MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>,
) -> MixedUnitMatrix<NewRowDims, NewColDims, nalgebra::OMatrix<T, R, C>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
    RowDims: RescaleFactors<NewRowDims>,
    ColDims: RescaleFactors<NewColDims>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
{
    let inner = matrix.nalgebra();
    let (r, c) = inner.shape_generic();
    let (nrows, ncols) = (r.value(), c.value());

    // Materialize the per-row and per-column scale factors from the type-level
    // dimension change. Each is O(n); they seed the O(n²) rescale below.
    let mut row_factors = vec![0.0f64; nrows];
    let mut col_factors = vec![0.0f64; ncols];
    <RowDims as RescaleFactors<NewRowDims>>::write_factors(&mut row_factors);
    <ColDims as RescaleFactors<NewColDims>>::write_factors(&mut col_factors);

    let out = nalgebra::OMatrix::<T, R, C>::from_fn_generic(r, c, |i, j| {
        let factor = <T as num_traits::FromPrimitive>::from_f64(row_factors[i] / col_factors[j])
            .expect("rescale factor is representable in the storage type");
        inner[(i, j)].clone() * factor
    });
    MixedUnitMatrix::from_nalgebra(out)
}

impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::SimdComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// Returns the adjoint (conjugate transpose).
    ///
    /// The units behave exactly like [`transpose`](Self::transpose): the
    /// transposed row/column dimension lists are the reciprocals of the original
    /// column/row lists. Over a real scalar field it is the transpose; over ℂ it
    /// also conjugates each entry, which leaves the (real-exponent) units
    /// untouched.
    pub fn adjoint(
        &self,
    ) -> MixedUnitMatrix<
        Mapped<Reciprocal, ColDims>,
        Mapped<Reciprocal, RowDims>,
        nalgebra::OMatrix<T, C, R>,
        Brand,
    >
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, R>,
        ColDims: MapUnits<Reciprocal>,
        RowDims: MapUnits<Reciprocal>,
    {
        MixedUnitMatrix {
            inner: self.inner.adjoint(),
            _dims: PhantomData,
        }
    }
}

// The "transpose-multiply" fused products. Both are exactly `self.transpose() *
// rhs` (resp. `self.adjoint() * rhs`) computed without materializing the
// transpose, so they inherit transpose's *reciprocating* unit rule rather than a
// clean swap: `selfᵀ` sends `⟨RowDims, ColDims⟩` to `⟨1/ColDims, 1/RowDims⟩`, so
// the left operand of the product carries `1/RowDims` on its (contracted) rows —
// which is why `rhs` must line its rows up as `1/RowDims`, and the result lands
// in `⟨1/ColDims, rhs-cols⟩`. The classic `AᵀA` (Gram) is the case `rhs = self`,
// which only type-checks when `RowDims` is dimensionless — the tell that a raw
// `AᵀA` silently assumes the identity metric on the row space.
impl<RowDims, ColDims, Brand, T, R, C, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// `selfᵀ · rhs`, equivalent to `self.transpose() * rhs` but without forming
    /// the transpose.
    ///
    /// `rhs`'s rows must be `1/RowDims` (the transposed self's contracted axis),
    /// and the result lands in `⟨1/ColDims, ColDimsB⟩` — the same units as
    /// `self.transpose() * rhs`. (For the Gram product `AᵀA`, take `rhs = self`;
    /// it type-checks exactly when `RowDims` is dimensionless.)
    pub fn tr_mul<ColDimsB, C2, SB>(
        &self,
        rhs: &MixedUnitMatrix<
            Mapped<Reciprocal, RowDims>,
            ColDimsB,
            nalgebra::Matrix<T, R, C2, SB>,
            Brand,
        >,
    ) -> MixedUnitMatrix<Mapped<Reciprocal, ColDims>, ColDimsB, nalgebra::OMatrix<T, C, C2>, Brand>
    where
        RowDims: MapUnits<Reciprocal>,
        ColDims: MapUnits<Reciprocal>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, R, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, C2>,
    {
        MixedUnitMatrix::from_nalgebra(self.inner.tr_mul(&rhs.inner))
    }

    /// `selfᴴ · rhs`, equivalent to `self.adjoint() * rhs` but without forming the
    /// adjoint. Dimensionally identical to [`tr_mul`](Self::tr_mul) (conjugation
    /// leaves the real-exponent units untouched); over ℂ it conjugates `self`.
    pub fn ad_mul<ColDimsB, C2, SB>(
        &self,
        rhs: &MixedUnitMatrix<
            Mapped<Reciprocal, RowDims>,
            ColDimsB,
            nalgebra::Matrix<T, R, C2, SB>,
            Brand,
        >,
    ) -> MixedUnitMatrix<Mapped<Reciprocal, ColDims>, ColDimsB, nalgebra::OMatrix<T, C, C2>, Brand>
    where
        RowDims: MapUnits<Reciprocal>,
        ColDims: MapUnits<Reciprocal>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, R, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, C2>,
    {
        MixedUnitMatrix::from_nalgebra(self.inner.ad_mul(&rhs.inner))
    }
}

impl<RowDims, ColDims, Brand, T, D, S>
    MixedUnitMatrix<RowDims, ColDims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
{
    /// Returns the trace — the sum of the diagonal entries.
    ///
    /// The diagonal entry `M(i, i)` has unit `RowDims[i] / ColDims[i]`, and
    /// summing them is well-typed only when they coincide — the
    /// [`UniformDiag`] condition, the same gate as
    /// [`eigenvalues`](Self::eigenvalues). The result carries that shared
    /// diagonal unit `U`.
    pub fn trace(&self) -> Quantity<DiagUnit<RowDims, ColDims>, T, Brand>
    where
        RowDims: UniformDiag<ColDims>,
        Quantity<DiagUnit<RowDims, ColDims>, T, Brand>: FromRaw<T>,
    {
        <Quantity<DiagUnit<RowDims, ColDims>, T, Brand> as FromRaw<T>>::from_raw(self.inner.trace())
    }
}

// Integer matrix powers exist only for *endomorphisms*: `Aᵏ` unfolds to repeated
// `A · A`, which the matmul rule admits only when the input and output spaces
// coincide — i.e. `RowDims == ColDims == Dims`. Every product `A(i,k)·A(k,j)`
// then has unit `(Dims[i]/Dims[k])·(Dims[k]/Dims[j]) = Dims[i]/Dims[j]`, so each
// power stays in the same space (`A⁰` being the identity). This is exactly the
// discrete-time state-transition case `x_{k+n} = Φⁿ x_k`.
impl<Dims, Brand, T, D, S> MixedUnitMatrix<Dims, Dims, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D>,
    S: nalgebra::storage::StorageMut<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Raises the (endomorphism) matrix to the integer power `exp`, keeping its
    /// space: `Aᵏ : <Dims, Dims>` for all `k`, with `A⁰` the identity.
    pub fn pow(&self, exp: u32) -> MixedUnitMatrix<Dims, Dims, nalgebra::OMatrix<T, D, D>, Brand> {
        MixedUnitMatrix::from_nalgebra(self.inner.pow(exp))
    }

    /// The matrix exponential `exp(A) = Σ Aⁿ/n!`. Like [`pow`](Self::pow) it is
    /// defined only for an endomorphism (`<Dims, Dims>`), since the series adds
    /// `A⁰ = I` to every power, and keeps the result in that space.
    ///
    /// This is the continuous→discrete bridge `A_d = exp(A_c · dt)`. A continuous
    /// rate map is `<Dims/T, Dims>` and so is not an endomorphism; scaling by the
    /// timestep `dt` promotes it to `<Dims, Dims>` before exponentiating.
    pub fn exp(&self) -> MixedUnitMatrix<Dims, Dims, nalgebra::OMatrix<T, D, D>, Brand> {
        // `exp` is inherent to owned `OMatrix`, so `clone_owned` is the storage
        // conversion (not a redundant copy) for our generic `S`. It's an O(n²)
        // copy in front of an O(n³) routine that clones its input internally
        // anyway, so specializing the owned case would only add dupe.
        MixedUnitMatrix::from_nalgebra(self.inner.clone_owned().exp())
    }
}
