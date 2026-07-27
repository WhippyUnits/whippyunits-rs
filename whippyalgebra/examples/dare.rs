//! A unit-safe discrete-time algebraic Riccati equation (DARE) solver, applied
//! to a DC-motor position/velocity LQR.  The implementation is generic over the
//! units and shapes of the state and input spaces.
//!
//! The numerics are the structure-preserving doubling algorithm (SDA) — ported
//! from calcmogul's `dare_bench`, after Chu, Fan, Lin & Wang, Int. J. Control
//! 77:8 (2004).

use whippyalgebra::nalgebra::{
    MixedUnitMatrix, SMatrix, block_matrix, generic_block, generic_matrix, mixed_unit_matrix,
    unblock_matrix, zeros,
};
use whippyalgebra::{DivBy, MapUnits, Mapped, MulBy, Reciprocal, dims};
use whippyunits::api::UnitDisplayExt;
use whippyunits::{qty, quantity, unit};

/// The dual (element-wise reciprocal) of a dimension list, `1/D`.
type Dual<D> = Mapped<Reciprocal, D>;
/// Seconds as a `Unit`, for expressing the derivative space `Statė = State / s`.
type Sec = unit!(s);
/// The derivative (element-wise `÷ s`) of a dimension list — the row/output
/// space of a continuous rate map, whose codomain is a time derivative.
type Deriv<D> = Mapped<DivBy<Sec>, D>;

/// A state endomorphism `State → State` (`A`, the doubling residual `W`, `Aₖ`).
type StateMap<State, const N: usize> = MixedUnitMatrix<State, State, SMatrix<f64, N, N>>;
/// The input matrix `B: Input → State`.
type InputMap<State, Input, const N: usize, const M: usize> =
    MixedUnitMatrix<State, Input, SMatrix<f64, N, M>>;
/// A state cost `State → State*` (`Q`, the running `Hₖ`, and the solution `P`).
type StateCost<State, const N: usize> = MixedUnitMatrix<Dual<State>, State, SMatrix<f64, N, N>>;
/// The input cost `R: Input → Input*`.
type InputCost<Input, const M: usize> = MixedUnitMatrix<Dual<Input>, Input, SMatrix<f64, M, M>>;
/// The control "grammian" `G = BR⁻¹Bᵀ: State* → State` (and the running `V₂`).
type Grammian<State, const N: usize> = MixedUnitMatrix<State, Dual<State>, SMatrix<f64, N, N>>;
/// The optimal feedback gain `K: State → Input`, so `u = −Kx` lands in the input
/// space.
type Gain<State, Input, const N: usize, const M: usize> =
    MixedUnitMatrix<Input, State, SMatrix<f64, M, N>>;

/// A discrete-time LQR problem `x_{k+1} = A x + B u` with cost `Σ xᵀQx + uᵀRu`,
/// generic over the state/input units (`State`, `Input`) and shapes (`N`, `M`).
/// Bundling the four matrices lets the solve and gain methods share one
/// parameterization.
struct DiscreteLqr<State, Input, const N: usize, const M: usize>
where
    State: MapUnits<Reciprocal>,
    Input: MapUnits<Reciprocal>,
{
    /// Transition `A: State → State`.
    a: StateMap<State, N>,
    /// Input `B: Input → State`.
    b: InputMap<State, Input, N, M>,
    /// State cost `Q: State → State*`.
    q: StateCost<State, N>,
    /// Input cost `R: Input → Input*`.
    r: InputCost<Input, M>,
}

// The generic_matrix attribute macro inserts the where-clauses necessary for use of unit-safe
// matrices in a generic context.
#[generic_matrix(
    matrix(rows(N, State), cols(N, State)),
    matrix(rows(N, State), cols(M, Input)),
    matrix(rows(M, Input), cols(M, Input))
)]
impl<State, Input, const N: usize, const M: usize> DiscreteLqr<State, Input, N, M>
where
    State: MapUnits<Reciprocal>,
    Input: MapUnits<Reciprocal>,
    Dual<State>: MapUnits<Reciprocal, Out = State>,
    Dual<Input>: MapUnits<Reciprocal, Out = Input>,
{
    /// Exact zero-order-hold discretization by the Van Loan augmented-matrix
    /// method.
    // The generic_block attribute macro inserts the where-clauses necessary for use of unit-safe
    // block matrix constructions in a generic context.
    #[generic_block(
        rows(N, State; M, Input),
        cols(N, State; M, Input),
    )]
    fn discretize<StateDot>(
        a_cont: MixedUnitMatrix<StateDot, State, SMatrix<f64, N, N>>,
        b_cont: MixedUnitMatrix<StateDot, Input, SMatrix<f64, N, M>>,
        dt: qty!(s),
    ) -> (StateMap<State, N>, InputMap<State, Input, N, M>)
    where
        // For exp to go through, we need to assert that multiplying the derivative space by
        // the timestep dt results in the original state space, so that the augmented matrix
        // is an endomorphism.
        StateDot: MapUnits<MulBy<Sec>, Out = State>,
    {
        let m_dt = block_matrix![
            [a_cont * dt, b_cont * dt],
            [zeros![Input, State], zeros![Input, Input]],
        ];

        let phi = m_dt.exp();

        unblock_matrix!(phi => [
            [a_d(N, N), b_d(N, M)],
            [_(M, N), _(M, M)],
        ]);
        (a_d, b_d)
    }

    /// Solves `P = AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q` by structure-preserving
    /// doubling. Returns the stabilizing solution `P: ⟨1/State, State⟩`.
    fn riccati_solution(&self) -> StateCost<State, N> {
        let r_inv = self.r.try_inverse().expect("R must be invertible");
        let b_t = self.b.transpose();
        let mut g_k: Grammian<State, N> = self.b * (r_inv * b_t); // G₀ = B R⁻¹ Bᵀ
        let mut h_k: StateCost<State, N> = self.q; // H₀ = Q
        let mut a_k: StateMap<State, N> = self.a; // A₀ = A

        let identity: StateMap<State, N> =
            StateMap::<State, N>::new(SMatrix::<f64, N, N>::identity());

        for _ in 0..100 {
            let w: StateMap<State, N> = identity + g_k * h_k; // W = I + Gₖ Hₖ
            let v1: StateMap<State, N> = w.solve(&a_k).expect("W must be invertible"); // W V₁ = Aₖ
            let g_k_t = g_k.transpose();
            let v2: Grammian<State, N> = w.solve(&g_k_t).expect("W must be invertible").transpose();

            let g_next: Grammian<State, N> = g_k + a_k * v2 * a_k.transpose();
            let h_next: StateCost<State, N> = h_k + v1.transpose() * h_k * a_k;
            let a_next: StateMap<State, N> = a_k * v1;

            // Stop on a dimensionless ratio of Frobenius norms — the one place a
            // reduction over a heterogeneous matrix is needed, so it drops to raw.
            let delta = (h_next - h_k).nalgebra().norm();
            let scale = h_next.nalgebra().norm().max(1.0);

            g_k = g_next;
            h_k = h_next;
            a_k = a_next;

            if delta <= 1e-12 * scale {
                break;
            }
        }

        h_k
    }

    /// Optimal gain `K = (R + BᵀPB)⁻¹ BᵀPA : ⟨Input, State⟩` for the closed-loop
    /// `u = −Kx`.
    fn gain(&self) -> Gain<State, Input, N, M> {
        let p = self.riccati_solution();
        let b_t = self.b.transpose();
        let r_bpb_inv = (self.r + b_t * p * self.b)
            .try_inverse()
            .expect("R + BᵀPB invertible");
        r_bpb_inv * (b_t * p * self.a)
    }
}

/// State space `State = [rot, rot/s]` (position, velocity).
type MotorState = dims![rot, rot / s];
/// Input space `Input = [V]` (drive voltage).
type MotorInput = dims![V];

fn main() {
    // DC-motor feedforward constants and loop period, dimensioned so the whole
    // discretization is unit-checked.
    let kv = quantity!(0.02, V * s / rot); // voltage per unit velocity
    let ka = quantity!(0.005, V * s ^ 2 / rot); // voltage per unit acceleration
    let dt = quantity!(0.02, s); // 50 Hz control loop

    let alpha = kv / ka; // kV/kA : 1/s
    let a_cont = mixed_unit_matrix![Deriv<MotorState>, MotorState;
        [quantity!(0.0, 1 / s), 1.0],        // ṗ = v
        [quantity!(0.0, 1 / s ^ 2), -alpha], // v̇ = −α v
    ];
    let b_cont = mixed_unit_matrix![Deriv<MotorState>, MotorInput;
        [quantity!(0.0, rot / V.s)], // position has no direct drive
        [1.0 / ka],                    // v̇ += (1/kA) u
    ];
    let (a, b) = DiscreteLqr::discretize(a_cont, b_cont, dt);

    let q = mixed_unit_matrix![Dual<MotorState>, MotorState;
        [quantity!(1.0, 1 / rot^2), quantity!(0.0, s / rot^2)], // row 0
        [quantity!(0.0, s / rot^2), quantity!(1.0, s^2 / rot^2)], // row 1
    ];
    let r = mixed_unit_matrix![Dual<MotorInput>, MotorInput;
        [quantity!(1.0, 1 / V^2)],
    ];

    let lqr = DiscreteLqr { a, b, q, r };

    let p = lqr.riccati_solution();
    let p00: qty!(1 / rot ^ 2) = p.get::<0, 0>();
    let p01: qty!(s / rot ^ 2) = p.get::<0, 1>();
    let p11: qty!(s ^ 2 / rot ^ 2) = p.get::<1, 1>();
    println!("\nRiccati solution P (⟨1/State, State⟩):");
    println!("  P[0,0] = {}", p00.unit_display());
    println!("  P[0,1] = {}", p01.unit_display());
    println!("  P[1,1] = {}", p11.unit_display());

    println!("\nOptimal gain K = [kp, kd] (u = -Kx, u in V):");
    println!("{}", lqr.gain());

    // Plug P back into the DARE; the residual is unit-checked as ⟨1/State, State⟩
    // and only its norm drops to raw (should be ~machine-epsilon).
    let b_t = b.transpose();
    let mid = (a.transpose() * p * b)
        * (r + b_t * p * b).try_inverse().expect("invertible")
        * (b_t * p * a);
    let residual = a.transpose() * p * a - p - mid + q;
    println!(
        "\nDARE residual ‖AᵀPA - P - AᵀPB(R+BᵀPB)⁻¹BᵀPA + Q‖ = {:e}",
        residual.nalgebra().norm()
    );
}
