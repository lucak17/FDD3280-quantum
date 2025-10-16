
#
# --- Configuration
#

BCS = 0.0
N_GRID_POINTS = 24
LAMBDA = 0.25
STEPS = 2
SHOT_LIST = [100, 500, 2000]
#SHOT_LIST = [1500]
REPETITIONS = 2
NOISE = True
#NOISE = False


SINGLE_STEP = True
BRANCHING_CIRCUIT_V0 = True
BRANCHING_UCRY = False

import os
import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UCRYGate
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeTorino

backend_ref = FakeBrisbane()
#backend_ref = FakeTorino()
noise_model = NoiseModel.from_backend(backend_ref)

print("Qiskit / Aer:", qiskit.__version__, qiskit_aer.__version__)


# ---------------------------
# Helpers from the paper
# ---------------------------
def angle_from_prob(p: float) -> float:
    """Return θ = 2*arcsin(sqrt(p)) with clipping to [0,1]."""
    p = float(np.clip(p, 0.0, 1.0))
    return 2.0 * np.arcsin(np.sqrt(p))

def diffusion_eq_one_step(u: np.ndarray, lam: float) -> np.ndarray:
    """Exact diffusion equation update for one time step with BCs u[0]=u[N+1]=0.""" 
    N = u.size - 2
    u_new = np.ones_like(u) * BCS
    for i in range(1, N+1):
        u_new[i] = lam * u[i-1] + (1.0 - 2.0*lam) * u[i] + lam * u[i+1]
    return u_new

#
# Quantum Simulator
#
def provide_simulator(noise=False):
    sim = AerSimulator.from_backend(backend_ref)
    sim.set_options(noise_model=None)
    if noise:
        sim.set_options(noise_model=noise_model)
    return sim



# ---------------------------
# Quantum branching micro-kernel
# ---------------------------
def branching_mk_circuit_V0(uL: float, uC: float, uR: float, 
                            wL: float, wC: float, wR: float, 
                            name: str = "node") -> QuantumCircuit:
    """
    Build the 3-qubit branching micro-kernel: s0 (selector root), s1 (left-branch selector), ro (readout).
    Leaves 00->L, 01->C, 10->R; leaf 11 unused.
    """

    # Selector angles
    theta0 = angle_from_prob(wR)    # sets P(s0=1) = wR
    left_mass = wL + wC
    thetaL = angle_from_prob(0.0 if left_mass == 0 else wC/left_mass)  # conditional P(s1=1 | s0=0) = wC/(wL+wC)

    # Leaf injection angles (values are probabilities in [0,1])
    phiL, phiC, phiR = angle_from_prob(uL), angle_from_prob(uC), angle_from_prob(uR)

    # Registers
    s0 = QuantumRegister(1, "s0")
    s1 = QuantumRegister(1, "s1")
    ro = QuantumRegister(1, "ro")
    c_out = ClassicalRegister(1, "c")
    qc = QuantumCircuit(s0, s1, ro, c_out, name=name)

    # --- Prepare selector superposition ---
    # Root split: P(s0=1) = wR
    qc.ry(theta0, s0[0])

    # Left-branch split on s1 when s0=0 (negative control via X - CRY - X)
    qc.x(s0[0])
    qc.cry(thetaL, s0[0], s1[0])
    qc.x(s0[0])

    # --- Leaf-addressed injections into readout ro ---
    # Leaf 00: controls s0=0 & s1=0  -> C^2 RY(phiL) on ro
    qc.x(s0[0]); qc.x(s1[0])
    qc.mcry(phiL, list([s0[0], s1[0]]), ro[0])
    #qc.append(RYGate(phiL).control(2), [s0[0], s1[0], ro[0]])
    qc.x(s1[0]); qc.x(s0[0])

    # Leaf 01: controls s0=0 & s1=1  -> C^2 RY(phiC) on ro
    qc.x(s0[0])
    qc.mcry(phiC, list([s0[0], s1[0]]), ro[0])
    qc.x(s0[0])

    # Leaf 10: control s0=1 (right branch unsplit) -> CRY(phiR) on ro
    qc.cry(phiR, s0[0], ro[0])
    
    # Measure the readout (ro)
    qc.measure(ro[0], c_out[0])
    return qc

def branching_mk_circuit_UCRY(uL: float, uC: float, uR: float,
                              wL: float, wC: float, wR: float,
                              name: str = "node_ucry") -> QuantumCircuit:
    """
    Branching micro-kernel with a multiplexed Ry on 'ro' instead of three leaf-addressed controls.
    Depth reduction: removes explicit CC-RY decompositions.
    """
    # Selector angles
    theta0 = angle_from_prob(wR)                       # P(s0=1) = wR
    left_mass = wL + wC
    thetaL = angle_from_prob(0.0 if left_mass == 0 else wC/left_mass)  # P(s1=1|s0=0)

    # Leaf injection angles
    phiL, phiC, phiR = angle_from_prob(uL), angle_from_prob(uC), angle_from_prob(uR)

    # Registers
    s0 = QuantumRegister(1, "s0")
    s1 = QuantumRegister(1, "s1")
    ro = QuantumRegister(1, "ro")
    c_out = ClassicalRegister(1, "c")
    qc = QuantumCircuit(s0, s1, ro, c_out, name=name)

    # Prepare selector superposition (use ctrl_state=0 to avoid X-...-X sandwiches)
    qc.ry(theta0, s0[0])  # root split
    # Left-branch split on s1 when s0=0 (negative control via X - CRY - X)
    qc.x(s0[0])
    qc.cry(thetaL, s0[0], s1[0])
    qc.x(s0[0])
    
    # --- leaf injections via a single multiplexed Ry ---
    # Controls order [s0, s1] → angles map to bitstrings 00,01,10,11 respectively.
    angles = [phiL, phiC, phiR, 0.0]  # 11 leaf has zero weight
    qc.append(UCRYGate(angles), [s0[0], s1[0], ro[0]])

    # Readout
    qc.measure(ro[0], c_out[0])
    return qc

def branching_mk_circuit_V1a(uL: float, uC: float, uR: float,
                             wL: float, wC: float, wR: float,
                             name: str = "node_anc") -> QuantumCircuit:
    from qiskit.circuit.library import RYGate
    s0 = QuantumRegister(1, "s0")
    s1 = QuantumRegister(1, "s1")
    ro = QuantumRegister(1, "ro")
    a  = QuantumRegister(1, "anc")
    c_out = ClassicalRegister(1, "c")
    qc = QuantumCircuit(s0, s1, ro, a, c_out, name=name)

    theta0 = angle_from_prob(wR)
    left_mass = wL + wC
    thetaL = angle_from_prob(0.0 if left_mass == 0 else wC/left_mass)
    phiL, phiC, phiR = angle_from_prob(uL), angle_from_prob(uC), angle_from_prob(uR)

    # selectors (same as Fig. 6)
    qc.ry(theta0, s0[0])
    qc.x(s0[0])
    qc.cry(thetaL, s0[0], s1[0])
    qc.x(s0[0])

    # --- Leaf 00: s0=0 & s1=0 ---
    qc.x(s0[0]); qc.x(s1[0])
    qc.ccx(s0[0], s1[0], a[0])                  # compute predicate into ancilla
    qc.cry(phiL, a[0], ro[0])                   # inject
    qc.ccx(s0[0], s1[0], a[0])                  # uncompute
    qc.x(s1[0]); qc.x(s0[0])

    # --- Leaf 01: s0=0 & s1=1 ---
    qc.x(s0[0])
    qc.ccx(s0[0], s1[0], a[0])
    qc.cry(phiC, a[0], ro[0])
    qc.ccx(s0[0], s1[0], a[0])
    qc.x(s0[0])

    # --- Leaf 10: s0=1 ---
    qc.cry(phiR, s0[0], ro[0])

    qc.measure(ro[0], c_out[0])
    return qc


def one_step_diffusion_eq(u: np.ndarray, lam: float, shots: int, sim=None, branching_circuit=branching_mk_circuit_V0) -> np.ndarray:
    """Evaluate one diffusion eq step with the branching kernel across all interior nodes."""
    if sim is None:
        sim = provide_simulator()

    N = u.size - 2
    wL, wC, wR = lam, 1.0 - 2.0*lam, lam
    # Build a batch of circuits (one per interior node)
    circs = [branching_circuit(u[i-1], u[i], u[i+1], wL, wC, wR, name=f"node_{i}")
             for i in range(1, N+1)]
    tcircs = transpile(circs, backend=sim, optimization_level=3)
    result = sim.run(tcircs, shots=shots).result()
    #result = sim.run(circs, shots=shots).result()
    u_new = np.ones_like(u) * BCS
    for i in range(1, N+1):
        counts = result.get_counts(i-1)
        u_new[i] = counts.get('1', 0) / shots
    return u_new

def list_backend_ops(backend):
    """Return a sorted list of operation/gate names supported by the backend."""
    
    cfg = backend.configuration()
    # Basis gates + common non-unitary ops if known
    ops = set(cfg.basis_gates)
    # Many backends still support measure/barrier by default
    ops.update({"measure", "barrier"})
    return sorted(ops)

def one_step_diffusion_eq_print_backend(u: np.ndarray, lam: float, shots: int, sim=None,
                                        branching_circuit=branching_mk_circuit_V0) -> np.ndarray:
    """Evaluate one diffusion step; build→transpile→execute each circuit sequentially.
       Prints backend gate set once and per-circuit transpiled depth/size."""
    if sim is None:
        sim = provide_simulator()

    # --- backend summary ---
    ops = list_backend_ops(sim)
    try:
        backend_name = sim.name() if callable(getattr(sim, "name", None)) else getattr(sim, "name", str(sim))
    except Exception:
        backend_name = str(sim)
    print(f"[Backend] {backend_name}")
    print(f"[Backend] Available operations/gates ({len(ops)}): {ops}")

    pmA = generate_preset_pass_manager(optimization_level=3, backend=backend_ref)

    # --- params ---
    N = u.size - 2
    wL, wC, wR = lam, 1.0 - 2.0*lam, lam

    # --- run each node independently ---
    u_new = np.ones_like(u) * BCS
    for i in range(1, N+1):
        # Build
        circ = branching_circuit(u[i-1], u[i], u[i+1], wL, wC, wR, name=f"node_{i}")

        # Transpile
        #tcirc = transpile(circ, backend=sim, optimization_level=3)
        #depth = tcirc.depth() or 0
        #size  = tcirc.size()
        #print(f"[Transpile] {tcirc.name}: depth={depth}, size={size}")

        tcirc = pmA.run(circ)
        print("[Transpile] - A ops:", tcirc.count_ops(), "depth:", tcirc.depth(), "size:", tcirc.size())
        # Execute
        result = sim.run(tcirc, shots=shots).result()
        counts = result.get_counts()

        # Update field (assumes single classical readout bit 'c')
        u_new[i] = counts.get('1', 0) / shots

    return u_new

# ---------------------------
# Experiments
# ---------------------------
def single_step_test(N=15, lam=0.25, shots_list=(4000, 30000),
                     one_step_circuit=one_step_diffusion_eq,
                     branching_circuit=branching_mk_circuit_V0,
                     noise=False):
    """
    Single step on N nodes
    """
    sim = provide_simulator(noise=noise)
    # Grid + IC: u(x,0) = sin(pi x), Dirichlet boundaries = 0
    x = np.linspace(0, 1, N+2)               # include ghost cells
    u0 = np.zeros(N+2) + BCS
    u0[1:-1] = np.sin(np.pi * x[1:-1]) + BCS
    # Exact discrete FTCS target after one step
    u_exact = diffusion_eq_one_step(u0, lam)

    fig1 = plt.figure()
    plt.plot(x[:], u_exact[:], label="Exact FTCS (1 step)")
    # Run branching kernel for each shot budget
    for M in shots_list:
        u_est = one_step_circuit(u0, lam, shots=M, sim=sim,
                                 branching_circuit=branching_mk_circuit_V0)
        plt.plot(x[:], u_est[:], marker='o', linestyle='--', label=f"Branching MK (M={M})")
    plt.xlabel("x")
    plt.ylabel("u after 1 step")
    plt.title(f"Single-step diffusion (N={N}, λ={lam})")
    plt.legend()
    plt.tight_layout()
    name_fig = f"diffusion_steps{0}_noise{noise}_circuit{branching_circuit}"
    fig1.savefig(name_fig, dpi=400, bbox_inches="tight")
    plt.show()


def convergence_test(N=65, lam=0.25, steps=47, shots_list=(1000,4000,8000,16000,32000), repetitions=10, nu=1.0,
                     one_step_circuit=one_step_diffusion_eq,
                     branching_circuit=branching_mk_circuit_V0,
                     noise=False):
    
    sim = provide_simulator(noise=noise)
    #sim = AerSimulator()
    dx = 1.0 / (N + 1)
    dt = lam * dx * dx / nu
    T = steps * dt

    x = np.linspace(0, 1, N+2)
    u_analytic = np.zeros(N+2) + BCS
    u_analytic[1:-1] = np.exp(-nu * (np.pi**2) * T) * np.sin(np.pi * x[1:-1]) + BCS

    fx = np.pi * (x[:] - 0.5) * 4
    kk = nu * (np.pi**2) * 2
    erro_analytic = np.sin( fx ) / ( fx ) * (1 - np.exp(- kk * T) )
    u_analytic_corrected = u_analytic - erro_analytic

    #plt.figure()
    #plt.plot(x[:], erro_analytic[:], label="Error")
    #plt.xlabel("x")
    #plt.ylabel("error(x, T)")
    #plt.legend()
    #plt.grid()
    #plt.tight_layout()
    #plt.show()

    # Print a header
    print(f"\nConvergence test: N={N}, steps={steps}, λ={lam}, ν={nu}, dx={dx:.6g}, dt={dt:.6g}, T={T:.6g}")

    # Collect final profiles for plotting for the *first* repetition only (illustrative)
    profiles_for_plot = {}

    # Error table as in the paper
    print("\nshots M    L_inf (mean ± std)    L2 (mean ± std)")
    for M in shots_list:
        L_inf = []
        L2 = []
        for r in range(repetitions):
            # Initial condition u(x,0)=sin(pi x), enforce Dirichlet boundaries
            u = np.zeros(N+2) + BCS
            u[1:-1] = np.sin(np.pi * x[1:-1]) + BCS

            # March 'steps' times using the branching micro-kernel
            for _ in range(steps):
                u = one_step_circuit(u, lam, shots=M, sim=sim,
                                 branching_circuit=branching_mk_circuit_V0)

            # Errors vs analytic PDE solution at time T
            diff = u[1:-1] - u_analytic[1:-1]
            linf = np.max(np.abs(diff))
            l2 = np.sqrt(np.mean(diff**2))
            L_inf.append(linf)
            L2.append(l2)

            if r == 0:
                profiles_for_plot[M] = u.copy()

        print(f"{M:<9d} {np.mean(L_inf):.4f} ± {np.std(L_inf):.4f}    {np.mean(L2):.4f} ± {np.std(L2):.4f}")

    # Plot analytic vs one representative sampling profile for each M
    fig1 = plt.figure()
    plt.plot(x[:], u_analytic[:], label="Analytic (sin IC)")
    plt.plot(x[:], u_analytic_corrected[:], label="Analytic corrected (sin IC)")
    for M in shots_list:
        plt.plot(x[:], profiles_for_plot[M][:], linestyle='--', marker="x", label=f"Sampling (M={M})")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title(f"Heat equation sampling convergence (T={T:.4g})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    name_fig = f"diffusion_steps{steps}_noise{noise}_circuit{branching_circuit}"
    fig1.savefig(name_fig, dpi=400, bbox_inches="tight")
    plt.show()

# ---------------------------
# Run the experiments
# ---------------------------
if __name__ == "__main__":

    #Single-step
    if SINGLE_STEP:
        single_step_test(N=N_GRID_POINTS, lam=LAMBDA, shots_list=SHOT_LIST, 
                        one_step_circuit=one_step_diffusion_eq_print_backend,
                        branching_circuit=branching_mk_circuit_V0, noise=NOISE)

    #Convergence, print L_inf/L2 stats
    if BRANCHING_CIRCUIT_V0:
        convergence_test(N=N_GRID_POINTS, lam=LAMBDA, steps=STEPS,
                        shots_list=SHOT_LIST,
                        repetitions=REPETITIONS, nu=1.0, 
                        one_step_circuit=one_step_diffusion_eq,
                        branching_circuit=branching_mk_circuit_V0, noise=NOISE)

    if BRANCHING_UCRY:
        convergence_test(N=N_GRID_POINTS, lam=LAMBDA, steps=STEPS,
                shots_list=SHOT_LIST,
                repetitions=REPETITIONS, nu=1.0, 
                one_step_circuit=one_step_diffusion_eq,
                branching_circuit=branching_mk_circuit_UCRY, noise=NOISE)