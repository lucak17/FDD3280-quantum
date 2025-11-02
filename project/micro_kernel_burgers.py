import os
import numpy as np, matplotlib.pyplot as plt

BCS = 0.0
N_GRID_POINTS = 64
LAMBDA = 0.25
NU_BURGER = 0.01/np.pi
STEPS = 40
SHOT_LIST =  [512, 1024, 2048]
BATCH_SIZE = 1
REPETITIONS = 3

USE_FAKE_HARDWARE = True
NOISE = False

import qiskit, qiskit_aer
from qiskit.circuit import Instruction
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


print("Qiskit / Aer:", qiskit.__version__, qiskit_aer.__version__)

#
# Quantum Simulator
#
backend_ref = FakeBrisbane()
noise_model = NoiseModel.from_backend(backend_ref)
def provide_simulator(noise=False):
    if USE_FAKE_HARDWARE == False:
        return AerSimulator()
    else:
        sim = AerSimulator.from_backend(backend_ref)
        sim.set_options(noise_model=None)
        if noise:
            sim.set_options(noise_model=noise_model)
    return sim

def angle_from_prob(p: float) -> float:
    """Return θ = 2*arcsin(sqrt(p)) with clipping to [0,1]."""
    p = float(np.clip(p, 0.0, 1.0))
    return 2.0 * np.arcsin(np.sqrt(p))

# ---------------------------
# Burgers' equation helpers
# ---------------------------

def burgers_convex_weights(u_i, dt, dx, nu):
    """
    Node-local convex weights for 1D viscous Burgers
    """
    lam_nu = nu * dt / (dx * dx)
    c = u_i * dt / dx
    cp = max(c, 0.0)
    cm = max(-c, 0.0)

    wL = lam_nu + cp
    wR = lam_nu + cm
    wC = 1.0 - (wL + wR)
    wL = max(0.0, wL)
    wR = max(0.0, wR)
    wC = max(0.0, wC)
    s = wL + wC + wR
    if s > 0:
        wL, wC, wR = wL / s, wC / s, wR / s
    else:
        wL, wC, wR = 0.0, 1.0, 0.0
    return wL, wC, wR

def choose_dt_burgers(u: np.ndarray, dx: float, nu: float, cfl: float = 0.7) -> float:
    """
    Choose a stable dt for Burgers + diffusion using the combined CFL:
        |u_i| * dt/dx + 2*nu*dt/dx^2 <= 1  for all i
        dt_adv <= dx / max|u|
        dt_diff<= (dx^2) / (2*nu)
    """
    umax = float(np.max(np.abs(u)))
    dt_adv = np.inf if umax == 0.0 else dx / umax
    dt_diff = np.inf if nu == 0.0 else (dx * dx) / (2.0 * nu)
    dt = cfl * min(dt_adv, dt_diff)
    return dt

def burgers_eq_one_step_classical(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """
    Deterministic explicit step using the same convex weights per node.
    Dirichlet ghost cells u[0]=u[N+1]=BCS are enforced as in your diffusion reference.
    """
    N = u.size - 2
    u_new = np.ones_like(u) * BCS
    for i in range(1, N+1):
        wL, wC, wR = burgers_convex_weights(u[i], dt, dx, nu)
        u_new[i] = wL * u[i-1] + wC * u[i] + wR * u[i+1]
    return u_new

# ---------------------------
# Quantum branching micro-kernel
# ---------------------------
def branching_mk_instruction(uL, uC, uR, wL, wC, wR, name="node") -> Instruction:
    # angles
    theta0 = angle_from_prob(wR)
    left_mass = wL + wC
    thetaL = angle_from_prob(wC / left_mass)
    phiL, phiC, phiR = angle_from_prob(uL), angle_from_prob(uC), angle_from_prob(uR)

    # 3 qubits: [s0, s1, ro]
    sub = QuantumCircuit(3, name=name)

    sub.ry(theta0, 0)
    sub.x(0); sub.cry(thetaL, 0, 1); sub.x(0)

    sub.x(0); sub.x(1); sub.mcry(phiL, [0, 1], 2); sub.x(1); sub.x(0)
    sub.x(0); sub.mcry(phiC, [0, 1], 2); sub.x(0)
    sub.cry(phiR, 0, 2)

    return sub.to_instruction()


# ---------------------------
# Quantum branching micro-kernel for Burgers (batched)
# ---------------------------
def encode_signed_to_prob(u_val: float, M: float) -> float:
    """
    maps [-M, M] to [0, 1]
    """
    if M <= 0:
        return 0.5
    return float(np.clip(0.5 + 0.5 * (u_val / M), 0.0, 1.0))

def decode_prob_to_signed(p: float, M: float) -> float:
    return M * (2.0 * float(np.clip(p, 0.0, 1.0)) - 1.0)

def build_batch_burgers(u: np.ndarray, dt: float, dx: float, nu: float,
                        start_idx: int, batch_size: int, M: float,
                        name="batch_burgers"):
    N = u.size - 2
    nodes = list(range(start_idx, min(start_idx + batch_size, N + 1)))
    mm = len(nodes)

    q = QuantumRegister(3 * mm, "q")
    c = ClassicalRegister(mm, "c")
    circ = QuantumCircuit(q, c, name=name)

    for j, i in enumerate(nodes):
        s0 = 3 * j; s1 = s0 + 1; ro = s0 + 2

        # Weights from the ORIGINAL signed u
        wL, wC, wR = burgers_convex_weights(u[i], dt, dx, nu)

        # Encode signed values to probailities
        pL = encode_signed_to_prob(u[i-1], M)
        pC = encode_signed_to_prob(u[i],   M)
        pR = encode_signed_to_prob(u[i+1], M)

        inst = branching_mk_instruction(pL, pC, pR, wL, wC, wR, name=f"burgers_node_{i}")
        circ.append(inst, [q[s0], q[s1], q[ro]])

    ro_qubits = [q[3*j + 2] for j in range(mm)]
    circ.measure(ro_qubits, c[:mm])
    return circ, nodes



def one_step_burgers_eq_batched(u: np.ndarray, dt: float, dx: float, nu: float,
                                shots: int, batch_size: int, sim=None) -> np.ndarray:
    if sim is None:
        sim = provide_simulator()

    n_points = u.size - 2
    u_new = np.ones_like(u) * BCS

    # Global scale for this step
    maxBurgers = max(float(np.max(np.abs(u[1:-1]))), 1e-12)

    batches, idx_groups = [], []
    for start in range(1, n_points + 1, batch_size):
        bc, nodes = build_batch_burgers(u, dt, dx, nu, start, batch_size, maxBurgers,
                                        name=f"batch_burgers_{start}")
        batches.append(bc)
        idx_groups.append(nodes)

    tbatches = transpile(batches, backend=sim, optimization_level=3, seed_transpiler=42)
    result = sim.run(tbatches, shots=shots).result()

    def p1_from_counts(counts, m, j):
        total = 0
        for s, n in counts.items():
            s = s.zfill(m)
            if s[::-1][j] == '1':
                total += n
        return total / shots

    for k, nodes in enumerate(idx_groups):
        counts = result.get_counts(k)
        m = len(nodes)
        for j, i in enumerate(nodes):
            p_est = p1_from_counts(counts, m, j)   # probability in [0,1]
            u_new[i] = decode_prob_to_signed(p_est, maxBurgers)

    return u_new

def get_circuit_info(u: np.ndarray, dt: float, dx: float, nu: float, batch_size: int, sim=None):
    if sim is None:
        sim = provide_simulator()

    n_points = u.size - 2
    # Global scale for this step
    maxBurgers = max(float(np.max(np.abs(u[1:-1]))), 1e-12)

    batches, idx_groups = [], []
    for start in range(1, n_points + 1, batch_size):
        bc, nodes = build_batch_burgers(u, dt, dx, nu, start, batch_size, maxBurgers,
                                        name=f"batch_burgers_{start}")
        batches.append(bc)
        idx_groups.append(nodes)

    tbatches = transpile(batches, backend=sim, optimization_level=3, seed_transpiler=42)
    metrics = []
    for k, qc in enumerate(tbatches):
        # Basic info
        n_qubits  = qc.num_qubits
        n_clbits  = qc.num_clbits
        size      = qc.size()                 # total instructions
        depth     = qc.depth()                # includes measure/barrier
        depth_nom = qc.depth(                 # ignore measure/barrier
            filter_function=lambda inst: inst.operation.name not in ("measure", "barrier")
        )
        ops = qc.count_ops() 
        metrics.append({
            "batch": k,
            "name": qc.name,
            "nodes": list(idx_groups[k]),
            "n_qubits": n_qubits,
            "n_clbits": n_clbits,
            "size": size,
            "depth": depth,
            "depth_no_meas": depth_nom,
            "ops": dict(ops)
        })

    # Print metrics
    for m in metrics:
        print(
            f"{m['name']}: depth={m['depth']} (no-meas {m['depth_no_meas']}), "
            f"size={m['size']}, qubits={m['n_qubits']}, operations={m['ops']}")
        
def burgers_test(N=63, steps=40, shots_list=(1000, 4000, 8000), repetitions=3, nu=0.01/np.pi,
                 batch_size=BATCH_SIZE, noise=False):
    """
    Run Burgers on x in [-1, 1] with u(0,x) = -sin(pi x), u(t,±1)=0.
    """
    sim = provide_simulator(noise=noise)

    # grid on [-1,1] with ghost cells
    x = np.linspace(-1.0, 1.0, N + 2)
    dx = 2.0 / (N + 1)

    u = np.zeros(N + 2) + BCS
    u[1:-1] = -np.sin(np.pi * x[1:-1]) + BCS
    dt = choose_dt_burgers(u, dx, nu, cfl=0.9)
    get_circuit_info(u, dt, dx, nu, batch_size=BATCH_SIZE, sim=sim)

    print(f"\nBurgers test: N={N}, steps={steps}, ν={nu}, dx={dx:.6g}")

    profiles = {}
    L_inf_tot = []
    L2_tot = []
    for ss in shots_list:
        L2 = []
        L_inf = []
        for r in range(repetitions):
            # Initial condition and boundaries
            u = np.zeros(N + 2) + BCS
            u[1:-1] = -np.sin(np.pi * x[1:-1]) + BCS
            u_ref = u.copy()
            dt = choose_dt_burgers(u_ref, dx, nu, cfl=0.9)
            print(f"Burgers: shots={ss}, rep={r+1}/{repetitions}")
            for n in range(steps):
                print("Step ", n, "/",steps)
                # Classical explicit update
                u_ref = burgers_eq_one_step_classical(u_ref, dt, dx, nu)
                # quantum
                u = one_step_burgers_eq_batched(u, dt, dx, nu, shots=ss, batch_size=batch_size, sim=sim)

            # errors
            diff = u[1:-1] - u_ref[1:-1]
            linf = np.max(np.abs(diff))
            l2 = np.sqrt(np.mean(diff**2))
            L_inf.append(linf)
            L2.append(l2)
            if r == 0:
                profiles[ss] = (u.copy(), u_ref.copy())
        L_inf_tot.append(L_inf)
        L2_tot.append(L2)
        print(f"{ss:<9d} {np.mean(L_inf):.4f} ± {np.std(L_inf):.4f}  {np.mean(L2):.4f} ± {np.std(L2):.4f}")

    print("\nBurgers' eq - errors")
    print("\nshots M    L_inf (mean ± std)    L2 (mean ± std)")
    for i, ss in enumerate(shots_list):
        L_inf = L_inf_tot[i]
        L2 = L2_tot[i]
        print(f"{ss:<9d} {np.mean(L_inf):.4f} ± {np.std(L_inf):.4f}  {np.mean(L2):.4f} ± {np.std(L2):.4f}")
    
    # Plot
    fig = plt.figure()
    u_mc, u_ref = profiles[shots_list[0]]
    plt.plot(x[:], u_ref[:], label=f"classical")
    for M in shots_list:
        u_mc, u_ref = profiles[M]
        plt.plot(x[:], u_mc[:], linestyle='--', marker="x", label=f"quantum (M={M})")
    plt.xlabel("x"), plt.ylabel("u(x, T)")
    plt.title("Burgers: classical vs quantum")
    plt.legend(); plt.grid(); plt.tight_layout()
    name_fig = f"burgers_steps{steps}_batch{batch_size}_noise{noise}.pdf"
    fig.savefig(name_fig, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Start experiments")

    # --- Burgers ---
    burgers_test(N=N_GRID_POINTS, steps=STEPS,
                 shots_list=SHOT_LIST, repetitions=REPETITIONS,
                 nu=NU_BURGER, batch_size=BATCH_SIZE, noise=NOISE)
