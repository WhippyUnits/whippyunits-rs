#[allow(unused_imports)] // re-exported siblings, for intra-doc links
use crate::nalgebra::*;

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use whippyunits::quantity::{Quantity, Unit};
use whippyunits::{DivUnit, MulUnit, UnitDiv, UnitMul};

use super::UniformUnitMatrix;
use crate::dims::{Dimensionless, UnitRescale};
use crate::entry::FromRaw;
use crate::index::{Nat, PowUnit, ShapeIndex, UnitPow};

/// The angular unit returned by [`UniformUnitMatrix::angle`] — plain radians on
/// the identity scale, so the stored value is the radian measure `acos(…)`
/// produces, with no hidden conversion factor.
type Radians = whippyunits::unit!(rad);

// -----------------------------------------------------------------------------
// Matrix-level algebra (uniform specialization)
// -----------------------------------------------------------------------------

// The matrix product multiplies entry units: every summed term of
// `C(i,k) = Σⱼ A(i,j)·B(j,k)` has unit `Ua · Ub`, so the result is uniform with
// entry unit `Ua · Ub`. (Unlike the mixed product there is no shared-inner
// constraint: the single-unit contract carries no column/row labels to match.)
impl<Ua, Ub, Brand, MA, MB> Mul<UniformUnitMatrix<Ub, MB, Brand>>
    for UniformUnitMatrix<Ua, MA, Brand>
where
    MA: Mul<MB>,
    Ua: UnitMul<Ub>,
{
    type Output = UniformUnitMatrix<MulUnit<Ua, Ub>, <MA as Mul<MB>>::Output, Brand>;

    fn mul(self, rhs: UniformUnitMatrix<Ub, MB, Brand>) -> Self::Output {
        UniformUnitMatrix {
            inner: self.inner * rhs.inner,
            _unit: PhantomData,
        }
    }
}

impl<U, Brand, MA, MB> Add<UniformUnitMatrix<U, MB, Brand>> for UniformUnitMatrix<U, MA, Brand>
where
    MA: Add<MB>,
{
    type Output = UniformUnitMatrix<U, <MA as Add<MB>>::Output, Brand>;

    fn add(self, rhs: UniformUnitMatrix<U, MB, Brand>) -> Self::Output {
        UniformUnitMatrix {
            inner: self.inner + rhs.inner,
            _unit: PhantomData,
        }
    }
}

impl<U, Brand, MA, MB> Sub<UniformUnitMatrix<U, MB, Brand>> for UniformUnitMatrix<U, MA, Brand>
where
    MA: Sub<MB>,
{
    type Output = UniformUnitMatrix<U, <MA as Sub<MB>>::Output, Brand>;

    fn sub(self, rhs: UniformUnitMatrix<U, MB, Brand>) -> Self::Output {
        UniformUnitMatrix {
            inner: self.inner - rhs.inner,
            _unit: PhantomData,
        }
    }
}

impl<U, Brand, MA, MB> AddAssign<UniformUnitMatrix<U, MB, Brand>>
    for UniformUnitMatrix<U, MA, Brand>
where
    MA: AddAssign<MB>,
{
    fn add_assign(&mut self, rhs: UniformUnitMatrix<U, MB, Brand>) {
        self.inner += rhs.inner;
    }
}

impl<U, Brand, MA, MB> SubAssign<UniformUnitMatrix<U, MB, Brand>>
    for UniformUnitMatrix<U, MA, Brand>
where
    MA: SubAssign<MB>,
{
    fn sub_assign(&mut self, rhs: UniformUnitMatrix<U, MB, Brand>) {
        self.inner -= rhs.inner;
    }
}

impl<U, Brand, M> Neg for UniformUnitMatrix<U, M, Brand>
where
    M: Neg,
{
    type Output = UniformUnitMatrix<U, <M as Neg>::Output, Brand>;

    fn neg(self) -> Self::Output {
        UniformUnitMatrix {
            inner: -self.inner,
            _unit: PhantomData,
        }
    }
}

// Scalar-by-quantity multiplication multiplies the entry unit by the scalar's.
impl<U, Brand, T, R, C, S, Sq, Dq> Mul<Quantity<Unit<Sq, Dq>, T, Brand>>
    for UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    nalgebra::Matrix<T, R, C, S>: Mul<T>,
    U: UnitMul<Unit<Sq, Dq>>,
{
    type Output = UniformUnitMatrix<
        MulUnit<U, Unit<Sq, Dq>>,
        <nalgebra::Matrix<T, R, C, S> as Mul<T>>::Output,
        Brand,
    >;

    fn mul(self, rhs: Quantity<Unit<Sq, Dq>, T, Brand>) -> Self::Output {
        UniformUnitMatrix {
            inner: self.inner * rhs.unsafe_value,
            _unit: PhantomData,
        }
    }
}

// Scalar-by-quantity division divides the entry unit by the scalar's.
impl<U, Brand, T, R, C, S, Sq, Dq> Div<Quantity<Unit<Sq, Dq>, T, Brand>>
    for UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    nalgebra::Matrix<T, R, C, S>: Div<T>,
    U: UnitDiv<Unit<Sq, Dq>>,
{
    type Output = UniformUnitMatrix<
        DivUnit<U, Unit<Sq, Dq>>,
        <nalgebra::Matrix<T, R, C, S> as Div<T>>::Output,
        Brand,
    >;

    fn div(self, rhs: Quantity<Unit<Sq, Dq>, T, Brand>) -> Self::Output {
        UniformUnitMatrix {
            inner: self.inner / rhs.unsafe_value,
            _unit: PhantomData,
        }
    }
}

impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::SimdComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// Multiplies every entry by a real scalar, leaving the unit unchanged.
    pub fn scale(
        &self,
        factor: T::SimdRealField,
    ) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.scale(factor))
    }

    /// Divides every entry by a real scalar, leaving the unit unchanged.
    pub fn unscale(
        &self,
        factor: T::SimdRealField,
    ) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.unscale(factor))
    }
}

/// Rescales a uniform matrix to a different unit of the same dimension — the
/// [`UniformUnitMatrix`] analog of whippyunits'
/// [`rescale`](whippyunits::api::rescale()).
///
/// Because every entry shares the single unit `U`, this is one gauge change,
/// not a per-row/column one: the target `NewU` must have the same dimension as
/// `U` (only the scale differs — a different dimension has no [`UnitRescale`]
/// impl and is a compile error, exactly as scalar `rescale` rejects a dimension
/// mismatch), and every entry is multiplied by the single factor `U -> NewU`.
/// This is scalar `rescale` applied to the whole matrix at once; contrast
/// [`rescale_matrix`](crate::nalgebra::rescale_matrix), whose mixed matrix needs
/// a distinct factor per row and per column.
///
/// The target unit is read from the return type, like `rescale` reads its
/// annotated result:
///
/// ```
/// use whippyalgebra::nalgebra::{
///     rescale_uniform_matrix, uniform_unit_matrix, Const, OMatrix, UniformUnitMatrix,
/// };
/// use whippyunits::{quantity, qty, unit};
///
/// let v = uniform_unit_matrix![m / s;
///     [quantity!(1.0, m / s), quantity!(2.0, m / s)],
///     [quantity!(3.0, m / s), quantity!(4.0, m / s)],
/// ];
///
/// // Rescale every entry from m/s to mm/s (× 1000).
/// let mm: UniformUnitMatrix<unit!(mm / s), OMatrix<f64, Const<2>, Const<2>>> =
///     rescale_uniform_matrix(&v);
/// let e: qty!(mm / s) = mm.get(0, 0);
/// assert!((e.unsafe_value - 1000.0).abs() < 1e-9);
/// ```
pub fn rescale_uniform_matrix<U, NewU, Brand, T, R, C, S>(
    matrix: &UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>,
) -> UniformUnitMatrix<NewU, nalgebra::OMatrix<T, R, C>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
    U: UnitRescale<NewU>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
{
    let factor = <T as num_traits::FromPrimitive>::from_f64(<U as UnitRescale<NewU>>::FACTOR)
        .expect("rescale factor is representable in the storage type");
    UniformUnitMatrix::from_nalgebra(matrix.nalgebra().map(|x| x * factor.clone()))
}

impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Returns the transpose, which is uniform with the same entry unit.
    ///
    /// `Mᵀ(i, j) = M(j, i)` still carries unit `U`, so — unlike the mixed
    /// matrix, whose transpose reciprocates the row/column lists — a uniform
    /// transpose changes only the shape, not the unit.
    pub fn transpose(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, C, R>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, R>,
    {
        UniformUnitMatrix {
            inner: self.inner.transpose(),
            _unit: PhantomData,
        }
    }
}

impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::SimdComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// Returns the adjoint (conjugate transpose), uniform with the same entry
    /// unit `U`.
    ///
    /// Like the uniform [`transpose`](Self::transpose), a single shared unit
    /// survives the swap unchanged (the mixed adjoint reciprocates the row/column
    /// lists); over ℂ this additionally conjugates each entry, leaving `U`
    /// untouched.
    pub fn adjoint(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, C, R>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, R>,
    {
        UniformUnitMatrix {
            inner: self.inner.adjoint(),
            _unit: PhantomData,
        }
    }
}

// The "transpose-multiply" fused products. Because a uniform transpose keeps the
// entry unit `U` unchanged (no reciprocation, unlike the mixed case), `selfᵀ ·
// rhs` multiplies the two entry units exactly as the ordinary product does — so
// `tr_mul`/`ad_mul` land in `U · Ub`, needing only `U: UnitMul<Ub>`.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// `selfᵀ · rhs`, equivalent to `self.transpose() * rhs`. Both operands share
    /// the row axis; the result is uniform in `U · Ub`.
    pub fn tr_mul<Ub, C2, SB>(
        &self,
        rhs: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, R, C2, SB>, Brand>,
    ) -> UniformUnitMatrix<MulUnit<U, Ub>, nalgebra::OMatrix<T, C, C2>, Brand>
    where
        U: UnitMul<Ub>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, R, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, C2>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.tr_mul(&rhs.inner))
    }

    /// `selfᴴ · rhs`, equivalent to `self.adjoint() * rhs`; dimensionally the same
    /// as [`tr_mul`](Self::tr_mul) (conjugation leaves `U` untouched).
    pub fn ad_mul<Ub, C2, SB>(
        &self,
        rhs: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, R, C2, SB>, Brand>,
    ) -> UniformUnitMatrix<MulUnit<U, Ub>, nalgebra::OMatrix<T, C, C2>, Brand>
    where
        U: UnitMul<Ub>,
        C2: nalgebra::Dim,
        SB: nalgebra::storage::Storage<T, R, C2>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<C, C2>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.ad_mul(&rhs.inner))
    }
}

// Component-wise (Hadamard) product/quotient. A uniform matrix carries one unit,
// so — unlike the mixed [`component_mul`](crate::nalgebra::MixedUnitMatrix::component_mul),
// whose per-axis lists multiply element-wise — the whole result is uniform in a
// single transformed unit: `U · Ub` for the product, `U / Ub` for the quotient.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<T, R, C>,
{
    /// Component-wise product with another uniform matrix of the same shape. The
    /// result is uniform in `U · Ub`.
    pub fn component_mul<Ub, SB>(
        &self,
        rhs: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, R, C, SB>, Brand>,
    ) -> UniformUnitMatrix<MulUnit<U, Ub>, nalgebra::OMatrix<T, R, C>, Brand>
    where
        U: UnitMul<Ub>,
        T: Mul<T, Output = T>,
        SB: nalgebra::RawStorage<T, R, C>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.zip_map(&rhs.inner, |a, b| a * b))
    }

    /// Component-wise quotient with another uniform matrix of the same shape. The
    /// result is uniform in `U / Ub`.
    pub fn component_div<Ub, SB>(
        &self,
        rhs: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, R, C, SB>, Brand>,
    ) -> UniformUnitMatrix<DivUnit<U, Ub>, nalgebra::OMatrix<T, R, C>, Brand>
    where
        U: UnitDiv<Ub>,
        T: Div<T, Output = T>,
        SB: nalgebra::RawStorage<T, R, C>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.zip_map(&rhs.inner, |a, b| a / b))
    }
}

// Component-wise min/max. Both operands and the result share the single unit `U`
// — comparing entries of the *same* unit is the very thing uniformity makes
// sound (the mixed matrix, whose entries carry different units, cannot offer a
// well-typed element-wise order).
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar + nalgebra::SimdPartialOrd,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// The component-wise infimum (element-wise min) with another uniform matrix
    /// of the same unit and shape; the result stays in `U`.
    pub fn inf<SB>(
        &self,
        other: &UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, SB>, Brand>,
    ) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        SB: nalgebra::storage::Storage<T, R, C>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.zip_map(&other.inner, |a, b| a.simd_min(b)))
    }

    /// The component-wise supremum (element-wise max) with another uniform matrix
    /// of the same unit and shape; the result stays in `U`.
    pub fn sup<SB>(
        &self,
        other: &UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, SB>, Brand>,
    ) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        SB: nalgebra::storage::Storage<T, R, C>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.zip_map(&other.inner, |a, b| a.simd_max(b)))
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, D>,
    Quantity<U, T, Brand>: FromRaw<T>,
{
    /// Returns the trace — the sum of the diagonal entries — in the shared entry
    /// unit `U`.
    ///
    /// A uniform matrix needs none of the homogeneity check the mixed
    /// [`trace`](crate::nalgebra::MixedUnitMatrix::trace) demands: every diagonal entry
    /// already carries the one unit `U`, so their sum does too.
    pub fn trace(&self) -> Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.trace())
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------
//
// A reduction collapses the whole matrix to a scalar. The mixed
// [`MixedUnitMatrix`] must *withhold* these — summing or normalizing entries of
// *different* units has no single well-typed result (the escape-hatch `norm()`
// in the DARE example is exactly a heterogeneous reduction that has to drop to
// raw). A uniform matrix shares one unit `U`, so every reduction lands cleanly:
// the sum / mean / min / max / individual entries stay in `U`; the Frobenius
// `norm` roots a sum of squares back to `U` (its square sitting in `U²`); and a
// dot against another uniform matrix multiplies the two entry units.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// The sum of every entry, in the shared unit `U`.
    pub fn sum(&self) -> Quantity<U, T, Brand>
    where
        Quantity<U, T, Brand>: FromRaw<T>,
    {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.sum())
    }

    /// The arithmetic mean of every entry, in `U`.
    pub fn mean(&self) -> Quantity<U, T, Brand>
    where
        Quantity<U, T, Brand>: FromRaw<T>,
    {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.mean())
    }

    /// The Frobenius norm `√(Σ |M(i, j)|²)`, in `U`.
    ///
    /// The squares live in `U²` and the root brings it back to `U` — so, unlike
    /// a [mixed matrix](crate::nalgebra::MixedUnitMatrix) where a sum of squares *across
    /// heterogeneous units* has no unit at all (and `norm` must therefore be
    /// taken on the raw inner), the uniform norm is cleanly dimensioned. It is
    /// real-valued, hence over `T::RealField`.
    pub fn norm(&self) -> Quantity<U, T::RealField, Brand>
    where
        Quantity<U, T::RealField, Brand>: FromRaw<T::RealField>,
    {
        <Quantity<U, T::RealField, Brand> as FromRaw<T::RealField>>::from_raw(self.inner.norm())
    }

    /// Alias of [`norm`](Self::norm) (nalgebra's L2 magnitude).
    pub fn magnitude(&self) -> Quantity<U, T::RealField, Brand>
    where
        Quantity<U, T::RealField, Brand>: FromRaw<T::RealField>,
    {
        <Quantity<U, T::RealField, Brand> as FromRaw<T::RealField>>::from_raw(
            self.inner.magnitude(),
        )
    }

    /// The squared Frobenius norm `Σ |M(i, j)|²`, in `U²` — cheaper than
    /// [`norm`](Self::norm) (no root) and the natural home of a sum of squares.
    pub fn norm_squared(&self) -> Quantity<MulUnit<U, U>, T::RealField, Brand>
    where
        U: UnitMul<U>,
        Quantity<MulUnit<U, U>, T::RealField, Brand>: FromRaw<T::RealField>,
    {
        <Quantity<MulUnit<U, U>, T::RealField, Brand> as FromRaw<T::RealField>>::from_raw(
            self.inner.norm_squared(),
        )
    }

    /// Alias of [`norm_squared`](Self::norm_squared) (nalgebra's squared L2
    /// magnitude).
    pub fn magnitude_squared(&self) -> Quantity<MulUnit<U, U>, T::RealField, Brand>
    where
        U: UnitMul<U>,
        Quantity<MulUnit<U, U>, T::RealField, Brand>: FromRaw<T::RealField>,
    {
        <Quantity<MulUnit<U, U>, T::RealField, Brand> as FromRaw<T::RealField>>::from_raw(
            self.inner.magnitude_squared(),
        )
    }

    /// The dot (Frobenius inner) product `Σ M(i, j) · N(i, j)` with another
    /// uniform matrix of the same shape. Every summed term multiplies the two
    /// entry units, so the result is in `U · Ub`.
    pub fn dot<Ub, Sb>(
        &self,
        other: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, R, C, Sb>, Brand>,
    ) -> Quantity<MulUnit<U, Ub>, T, Brand>
    where
        U: UnitMul<Ub>,
        Sb: nalgebra::storage::Storage<T, R, C>,
        Quantity<MulUnit<U, Ub>, T, Brand>: FromRaw<T>,
    {
        <Quantity<MulUnit<U, Ub>, T, Brand> as FromRaw<T>>::from_raw(self.inner.dot(&other.inner))
    }

    /// The `p`-norm `(Σ |M(i, j)|ᵖ)^{1/p}`, in the shared unit `U`.
    ///
    /// Like the Frobenius [`norm`](Self::norm) the power and its root cancel back
    /// to a single `U`, so — unlike a heterogeneous [mixed
    /// matrix](crate::nalgebra::MixedUnitMatrix) — the result is cleanly dimensioned.
    pub fn lp_norm(&self, p: i32) -> Quantity<U, T::RealField, Brand>
    where
        Quantity<U, T::RealField, Brand>: FromRaw<T::RealField>,
    {
        <Quantity<U, T::RealField, Brand> as FromRaw<T::RealField>>::from_raw(self.inner.lp_norm(p))
    }

    /// Sum of every column, collapsed to a row vector (`1 × C`) in `U` —
    /// entry `j` is `Σᵢ M(i, j)`. Every summed term shares `U`, so the axis
    /// reduction stays cleanly dimensioned.
    pub fn row_sum(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, nalgebra::U1, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.row_sum())
    }

    /// Sum of every row, collapsed to a column vector (`R × 1`) in `U` —
    /// entry `i` is `Σⱼ M(i, j)`.
    pub fn column_sum(&self) -> UniformUnitMatrix<U, nalgebra::OVector<T, R>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.column_sum())
    }

    /// Mean of every column, as a row vector (`1 × C`) in `U`.
    pub fn row_mean(&self) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, nalgebra::U1, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.row_mean())
    }

    /// Mean of every row, as a column vector (`R × 1`) in `U`.
    pub fn column_mean(&self) -> UniformUnitMatrix<U, nalgebra::OVector<T, R>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.column_mean())
    }

    /// The (population) variance of every entry, in `U²` — a mean of squared
    /// deviations, so the unit squares. Cheaper than a norm and the natural
    /// scalar companion to [`mean`](Self::mean).
    pub fn variance(&self) -> Quantity<MulUnit<U, U>, T, Brand>
    where
        U: UnitMul<U>,
        Quantity<MulUnit<U, U>, T, Brand>: FromRaw<T>,
    {
        <Quantity<MulUnit<U, U>, T, Brand> as FromRaw<T>>::from_raw(self.inner.variance())
    }

    /// Per-column variance, as a row vector (`1 × C`) in `U²`.
    pub fn row_variance(
        &self,
    ) -> UniformUnitMatrix<MulUnit<U, U>, nalgebra::OMatrix<T, nalgebra::U1, C>, Brand>
    where
        U: UnitMul<U>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.row_variance())
    }

    /// Per-row variance, as a column vector (`R × 1`) in `U²`.
    pub fn column_variance(
        &self,
    ) -> UniformUnitMatrix<MulUnit<U, U>, nalgebra::OVector<T, R>, Brand>
    where
        U: UnitMul<U>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.column_variance())
    }

    /// The product of every entry, in `U^(R·C)`.
    ///
    /// Every one of the `R·C` entries shares `U`, so their product raises `U` to
    /// the entry count — a compile-time constant read from the (statically
    /// sized) shape via [`ShapeIndex`], then applied by [`UnitPow`]. This is only
    /// available at a static size: a `Dyn` count would live at runtime and have
    /// no type-level power.
    pub fn product(
        &self,
    ) -> Quantity<PowUnit<U, typenum::Prod<<R as ShapeIndex>::Nat, <C as ShapeIndex>::Nat>>, T, Brand>
    where
        R: ShapeIndex,
        C: ShapeIndex,
        <R as ShapeIndex>::Nat: core::ops::Mul<<C as ShapeIndex>::Nat>,
        U: UnitPow<typenum::Prod<<R as ShapeIndex>::Nat, <C as ShapeIndex>::Nat>>,
        Quantity<
            PowUnit<U, typenum::Prod<<R as ShapeIndex>::Nat, <C as ShapeIndex>::Nat>>,
            T,
            Brand,
        >: FromRaw<T>,
    {
        FromRaw::from_raw(self.inner.product())
    }
}

impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, D, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::DimMin<D, Output = D> + ShapeIndex,
    S: nalgebra::storage::Storage<T, D, D>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D, D> + nalgebra::allocator::Allocator<D>,
{
    /// Raises the matrix to the compile-time power `K`, landing in `Uᴷ`
    /// (`K = 0` is the dimensionless identity). The const exponent is what makes
    /// this typeable — a runtime `pow(k)` would have a runtime unit `Uᵏ`, so it
    /// exists only when `U` is dimensionless.
    pub fn powi<const K: usize>(
        &self,
    ) -> UniformUnitMatrix<PowUnit<U, Nat<K>>, nalgebra::OMatrix<T, D, D>, Brand>
    where
        ::nalgebra::Const<K>: ShapeIndex,
        U: UnitPow<Nat<K>>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.clone_owned().pow(K as u32))
    }
}

// ---------------------------------------------------------------------------
// Vector geometry
// ---------------------------------------------------------------------------
//
// These live only on uniform *column vectors* — a mixed vector's entries carry
// different units, so a norm / angle / interpolation over them has no single
// well-typed result. With one shared unit `U` the geometry lands cleanly:
//
// - `cross` is genuinely bilinear, so it multiplies the two entry units
//   (`U · Ub`), exactly like [`dot`](UniformUnitMatrix::dot);
// - `lerp` is an affine blend within one space, so it keeps `U` (the parameter
//   `t` is a dimensionless mixing weight);
// - `metric_distance` is `‖self − other‖`, so it keeps `U` — the operands must
//   already share `U`;
// - `angle` and `slerp` normalize their inputs, so the units cancel: `angle`
//   lands in dimensionless radians and `slerp` returns a dimensionless unit
//   vector, just like [`normalize`](UniformUnitMatrix::normalize).
//
// `angle`/`metric_distance`/`slerp` read the vectors through the *identity*
// inner product — the same silent Euclidean metric an [`svd`](UniformUnitMatrix::svd)
// uses — so the pair really has to be the same physical vector space (same `U`)
// for the answer to be honest.

/// The 3-D cross product lives on statically-3 column vectors. `cross` is
/// bilinear, so it multiplies the two entry units.
impl<U, Brand, T, S> UniformUnitMatrix<U, nalgebra::Matrix<T, nalgebra::U3, nalgebra::U1, S>, Brand>
where
    T: nalgebra::ComplexField,
    S: nalgebra::storage::RawStorage<T, nalgebra::U3, nalgebra::U1>,
{
    /// The 3-D cross product `self × other`, in `U · Ub`.
    pub fn cross<Ub, S2>(
        &self,
        other: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, nalgebra::U3, nalgebra::U1, S2>, Brand>,
    ) -> UniformUnitMatrix<MulUnit<U, Ub>, nalgebra::OMatrix<T, nalgebra::U3, nalgebra::U1>, Brand>
    where
        U: UnitMul<Ub>,
        S2: nalgebra::storage::RawStorage<T, nalgebra::U3, nalgebra::U1>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.cross(&other.inner))
    }
}

/// Affine interpolation between two vectors of the shared unit `U`.
impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S>, Brand>
where
    T: nalgebra::ComplexField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// Linear interpolation `self · (1 − t) + other · t`, keeping the shared
    /// unit `U`. `t` is a dimensionless mixing weight (not clamped to `[0, 1]`).
    pub fn lerp<S2>(
        &self,
        other: &UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S2>, Brand>,
        t: T,
    ) -> UniformUnitMatrix<U, nalgebra::OVector<T, D>, Brand>
    where
        S2: nalgebra::storage::Storage<T, D, nalgebra::U1>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.lerp(&other.inner, t))
    }
}

/// Real-valued geometry (angle, distance, spherical interpolation) needs an
/// ordered/real scalar, so it sits on `T: RealField`.
impl<U, Brand, T, D, S> UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S>, Brand>
where
    T: nalgebra::RealField,
    D: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, D, nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    /// The Euclidean angle between two vectors of the same unit `U`, in
    /// radians — the units cancel because `angle` normalizes both operands.
    pub fn angle<S2>(
        &self,
        other: &UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S2>, Brand>,
    ) -> Quantity<Radians, T, Brand>
    where
        S2: nalgebra::storage::Storage<T, D, nalgebra::U1>,
        Quantity<Radians, T, Brand>: FromRaw<T>,
    {
        <Quantity<Radians, T, Brand> as FromRaw<T>>::from_raw(self.inner.angle(&other.inner))
    }

    /// The distance `‖self − other‖` between two vectors of the same unit `U`,
    /// in `U`.
    pub fn metric_distance<S2>(
        &self,
        other: &UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S2>, Brand>,
    ) -> Quantity<U, T, Brand>
    where
        S2: nalgebra::storage::Storage<T, D, nalgebra::U1>,
        Quantity<U, T, Brand>: FromRaw<T>,
    {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.metric_distance(&other.inner))
    }

    /// Alias of [`metric_distance`](Self::metric_distance).
    pub fn distance<S2>(
        &self,
        other: &UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S2>, Brand>,
    ) -> Quantity<U, T, Brand>
    where
        S2: nalgebra::storage::Storage<T, D, nalgebra::U1>,
        Quantity<U, T, Brand>: FromRaw<T>,
    {
        self.metric_distance(other)
    }

    /// Spherical linear interpolation. The result is a dimensionless unit
    /// vector (`slerp` normalizes its inputs), so the units cancel exactly as in
    /// [`normalize`](Self::normalize).
    pub fn slerp<S2>(
        &self,
        other: &UniformUnitMatrix<U, nalgebra::Matrix<T, D, nalgebra::U1, S2>, Brand>,
        t: T,
    ) -> UniformUnitMatrix<Dimensionless, nalgebra::OVector<T, D>, Brand>
    where
        S2: nalgebra::storage::Storage<T, D, nalgebra::U1>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.slerp(&other.inner, t))
    }
}

// The order-based reductions need a genuinely ordered (real) scalar, so they
// sit on `T: RealField` rather than `ComplexField`. Each returns an actual
// entry (or its magnitude), so the unit is simply `U`.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::RealField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
    Quantity<U, T, Brand>: FromRaw<T>,
{
    /// The largest entry, in `U`.
    pub fn max(&self) -> Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.max())
    }

    /// The smallest entry, in `U`.
    pub fn min(&self) -> Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.min())
    }

    /// The largest entry by absolute value, in `U`.
    pub fn amax(&self) -> Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.amax())
    }

    /// The smallest entry by absolute value, in `U`.
    pub fn amin(&self) -> Quantity<U, T, Brand> {
        <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(self.inner.amin())
    }
}

impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar + Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
    Quantity<U, T, Brand>: FromRaw<T>,
{
    /// Applies a unit-preserving closure to every entry, returning a new uniform
    /// matrix of the same unit.
    ///
    /// The closure receives and returns a `Quantity` of the shared entry unit,
    /// so the unit is checked at the boundary and cannot silently change — the
    /// bulk-`map` the mixed matrix cannot offer.
    pub fn map<F>(&self, mut f: F) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        F: FnMut(Quantity<U, T, Brand>) -> Quantity<U, T, Brand>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(
            self.inner
                .map(|v| f(<Quantity<U, T, Brand> as FromRaw<T>>::from_raw(v)).unsafe_value),
        )
    }

    /// Combines this matrix with another of the same shape entry-by-entry,
    /// returning a new uniform matrix whose unit `Ur` is whatever the closure
    /// produces — the two-input sibling of [`map`](Self::map).
    ///
    /// The closure receives a `Quantity<U>` and a `Quantity<Ub>` and returns a
    /// `Quantity<Ur>`, so all three units are checked at the boundary. Unlike
    /// [`component_mul`](Self::component_mul) / [`component_div`](Self::component_div)
    /// (which fix `Ur` to `U · Ub` / `U / Ub`), the output unit here is free.
    pub fn zip_map<Ub, Ur, Sb, F>(
        &self,
        other: &UniformUnitMatrix<Ub, nalgebra::Matrix<T, R, C, Sb>, Brand>,
        mut f: F,
    ) -> UniformUnitMatrix<Ur, nalgebra::OMatrix<T, R, C>, Brand>
    where
        F: FnMut(Quantity<U, T, Brand>, Quantity<Ub, T, Brand>) -> Quantity<Ur, T, Brand>,
        Sb: nalgebra::storage::Storage<T, R, C>,
        Quantity<Ub, T, Brand>: FromRaw<T>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.zip_map(&other.inner, |a, b| {
            f(
                <Quantity<U, T, Brand> as FromRaw<T>>::from_raw(a),
                <Quantity<Ub, T, Brand> as FromRaw<T>>::from_raw(b),
            )
            .unsafe_value
        }))
    }
}

// Broadcast a same-unit scalar across every entry. Adding a `Quantity<U>` to a
// uniform matrix in `U` is well-typed (both sides share the unit) and leaves the
// result in `U` — the whole-matrix affine shift the mixed type cannot offer.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::Scalar + nalgebra::ClosedAddAssign,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// Adds a scalar `Quantity` (of the shared unit `U`) to every entry, staying
    /// in `U`.
    pub fn add_scalar(
        &self,
        scalar: Quantity<U, T, Brand>,
    ) -> UniformUnitMatrix<U, nalgebra::OMatrix<T, R, C>, Brand>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.add_scalar(scalar.unsafe_value))
    }
}

// Normalization divides every entry by the (same-unit) norm, so the result is
// the ratio `U / U` — dimensionless at every concrete unit. There is no
// `normalize_mut`: the unit *changes* (`U → U/U`), so it cannot be done in place.
impl<U, Brand, T, R, C, S> UniformUnitMatrix<U, nalgebra::Matrix<T, R, C, S>, Brand>
where
    T: nalgebra::ComplexField,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::storage::Storage<T, R, C>,
{
    /// The normalized matrix `M / ‖M‖` (unit Frobenius norm), in the quotient    /// unit `U / U` — i.e. dimensionless. The magnitude that was divided out is
    /// available from [`norm`](Self::norm).
    pub fn normalize(&self) -> UniformUnitMatrix<DivUnit<U, U>, nalgebra::OMatrix<T, R, C>, Brand>
    where
        U: UnitDiv<U>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        UniformUnitMatrix::from_nalgebra(self.inner.normalize())
    }

    /// Like [`normalize`](Self::normalize), but returns `None` when the norm is
    /// at or below `min_norm` (i.e. the matrix is effectively zero and has no
    /// well-defined direction). The result is dimensionless (`U / U`).
    pub fn try_normalize(
        &self,
        min_norm: T::RealField,
    ) -> Option<UniformUnitMatrix<DivUnit<U, U>, nalgebra::OMatrix<T, R, C>, Brand>>
    where
        U: UnitDiv<U>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R, C>,
    {
        self.inner
            .try_normalize(min_norm)
            .map(UniformUnitMatrix::from_nalgebra)
    }
}
