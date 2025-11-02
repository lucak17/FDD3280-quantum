# =========================
# Wave equation (1D, periodic) – classical + quantum
# =========================

import os
import numpy as np, matplotlib.pyplot as plt

BCS = 0.0
N_GRID_POINTS = 64
LAMBDA = 0.8
C_WAVE = 1.0  # wave speed c
STEPS = 30
SHOT_LIST =  [100000, 200000]
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


def encode_signed_to_prob(u_val: float, M: float) -> float:
    """
    maps [-M, M] to [0, 1]
    """
    if M <= 0:
        return 0.5
    return float(np.clip(0.5 + 0.5 * (u_val / M), 0.0, 1.0))

def decode_prob_to_signed(p: float, M: float) -> float:
    return M * (2.0 * float(np.clip(p, 0.0, 1.0)) - 1.0)

def wave_weights(lam: float):
    """
    Four-term affine weights in the leapfrog update:
      E^{n+1}_i = aL*E_{i-1}^n + aC*E_i^n + aR*E_{i+1}^n + aP*E_i^{n-1}
    where aL=aR=lam^2, aC=2-2 lam^2, aP=-1. Sum = 1.
    """
    aL = lam**2
    aR = lam**2
    aC = 2.0 - 2.0 * lam**2
    aP = -1.0
    return aL, aC, aR, aP

# ---------- Classical (periodic) ----------
def laplacian_periodic(E: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(E, -1) - 2.0*E + np.roll(E, 1)) / (dx*dx)

def wave_first_step_taylor(E0: np.ndarray, dx: float, dt: float, c: float) -> np.ndarray:
    """
    E^1 = E^0 + dt * (dt E)(x,0) + 0.5 * (c*dt)^2 * E_xx(x,0).
    Exact dt E(x,0) = -c*pi*cos(pi*x)
    """
    n_points = E0.size
    x = np.linspace(-1.0, 1.0, n_points, endpoint=False)
    V0 = -c * np.pi * np.cos(np.pi * x)
    return E0 + dt * V0 + 0.5 * (c*dt)**2 * laplacian_periodic(E0, dx)

def wave_eq_one_step_classical(E_prev: np.ndarray, E_curr: np.ndarray, lam: float) -> np.ndarray:
    """
    Leapfrog explicit update with periodic BCs
    """
    aL, aC, aR, aP = wave_weights(lam)
    E_next = ( aL * np.roll(E_curr,  1) + aC * E_curr +
                aR * np.roll(E_curr, -1) + aP * E_prev )
    return E_next

# ---------- Quantum branching micro-kernel for wave ----------
def branching4_mk_instruction(pL, pC, pR, pP, lam, name="wave4_node") -> Instruction:
    """
    One 3-qubit subcircuit (s0, s1, ro) that realizes a 4-outcome selector with
    normalized weights:
        wL = lam^2/3, wC = (2 - 2*lam^2)/3, wR = lam^2/3, wP = 1/3
    and injects leaf probs:
        L: pL,  C: pC,  R: pR,  P: (1 - pP) 
    The measured Pr(ro=1) equals wL*pL + wC*pC + wR*pR + wP*(1 - pP).
    """
    # normalized weights
    wL = (lam**2) / 3.0
    wC = (2.0 - 2.0*lam**2) / 3.0
    wR = (lam**2) / 3.0
    wP = 1.0 / 3.0
    # group masses for a two-level binary tree: A={L,C}, B={R,P}
    gA = wL + wC
    gB = wR + wP

    # angles for selector
    theta0 = angle_from_prob(gB)                          # sets Pr(s0=1) = gB
    thetaA = angle_from_prob(0.0 if gA == 0 else wC/gA)   # Pr(s1=1 | s0=0) = wC/gA
    thetaB = angle_from_prob(0.0 if gB == 0 else wP/gB)   # Pr(s1=1 | s0=1) = wP/gB

    # leaf injection angles
    phiL = angle_from_prob(pL)
    phiC = angle_from_prob(pC)
    phiR = angle_from_prob(pR)
    phiPc = angle_from_prob(1.0 - pP)

    # 3 qubits: [s0, s1, ro]
    sub = QuantumCircuit(3, name=name)
    # Top split: A={L,C} vs B={R,P}
    sub.ry(theta0, 0)
    # Conditional split within A when s0=0
    sub.x(0); sub.cry(thetaA, 0, 1); sub.x(0)
    # Conditional split within B when s0=1
    sub.cry(thetaB, 0, 1)

    # --- leaf injections into ro ---
    # L: (s0=0, s1=0)
    sub.x(0); sub.x(1); sub.mcry(phiL, [0, 1], 2); sub.x(1); sub.x(0)
    # C: (s0=0, s1=1)
    sub.x(0); sub.mcry(phiC, [0, 1], 2); sub.x(0)
    # R: (s0=1, s1=0)
    sub.x(1); sub.mcry(phiR, [0, 1], 2); sub.x(1)
    # P: (s0=1, s1=1)
    sub.mcry(phiPc, [0, 1], 2)

    return sub.to_instruction()


def build_batch_wave_1kernel(E_prev: np.ndarray, E_curr: np.ndarray, lam: float,
                             start_idx: int, batch_size: int, M: float,
                             name="batch"):
    """
    Batch circuit: 3 qubits per node (s0, s1, ro), 1 classical bit per node.
    Uses the 4-leaf microkernel above. Returns (QuantumCircuit, node_indices).
    """
    n_points = E_curr.size
    nodes = list(range(start_idx, min(start_idx + batch_size, n_points)))
    mm = len(nodes)

    q = QuantumRegister(3 * mm, "q")
    c = ClassicalRegister(mm, "c")
    circ = QuantumCircuit(q, c, name=name)

    for j, i in enumerate(nodes):
        # periodic BCs
        im1 = (i - 1 + n_points) % n_points
        ip1 = (i + 1) % n_points

        # Encode to probabilities with a single shared scale M
        pL = encode_signed_to_prob(E_curr[im1], M)
        pC = encode_signed_to_prob(E_curr[i],   M)
        pR = encode_signed_to_prob(E_curr[ip1], M)
        pP = encode_signed_to_prob(E_prev[i],   M)

        inst = branching4_mk_instruction(pL, pC, pR, pP, lam, name=f"wave4_{i}")
        s0 = 3*j; s1 = s0 + 1; ro = s0 + 2
        circ.append(inst, [q[s0], q[s1], q[ro]])

    # Measure only the readout qubits
    ro_qubits = [q[3*j + 2] for j in range(mm)]
    circ.measure(ro_qubits, c[:mm])
    return circ, nodes

def one_step_wave_eq_batched(E_prev: np.ndarray, E_curr: np.ndarray, lam: float,
                             shots: int, batch_size: int, sim=None) -> np.ndarray:
    if sim is None:
        sim = provide_simulator()

    n_points = E_curr.size
    maxE = max(float(np.max(np.abs(E_curr))), float(np.max(np.abs(E_prev))), 1e-12)
        
    batches, idx_groups = [], []
    for start in range(0, n_points, batch_size):
        bc, nodes = build_batch_wave_1kernel(E_prev, E_curr, lam, start, batch_size, maxE, name=f"batch_{start}")
        batches.append(bc); idx_groups.append(nodes)

    tbatches = transpile(batches, backend=sim, optimization_level=3, seed_transpiler=42)
    result = sim.run(tbatches, shots=shots).result()

    def p1_from_counts(counts, m, j):
        tot = 0
        for s, n in counts.items():
            s = s.zfill(m)
            if s[::-1][j] == '1':
                tot += n
        return tot / shots
    
    p_meas = np.zeros(n_points)
    for k, nodes in enumerate(idx_groups):
        counts = result.get_counts(k)
        m = len(nodes)
        for j, i in enumerate(nodes):
            p_meas[i] = p1_from_counts(counts, m, j)

    # E^{n+1} = M * (6*p_meas - 3)
    E_next = maxE * (6.0 * p_meas - 3.0)
    return E_next



def get_circuit_info(E_prev: np.ndarray, E_curr: np.ndarray, lam: float, batch_size: int, sim=None):
    if sim is None:
        sim = provide_simulator()

    n_points = E_curr.size
    maxE = max(float(np.max(np.abs(E_curr))), float(np.max(np.abs(E_prev))), 1e-12)
        
    batches, idx_groups = [], []
    for start in range(0, n_points, batch_size):
        bc, nodes = build_batch_wave_1kernel(E_prev, E_curr, lam, start, batch_size, maxE,name=f"batch_{start}")
        batches.append(bc); idx_groups.append(nodes)

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


def wave_test(N=N_GRID_POINTS, steps=STEPS, CFL=LAMBDA, c=C_WAVE,
              shots_list=SHOT_LIST, repetitions=REPETITIONS,
              batch_size=BATCH_SIZE, noise=NOISE):
    """
    Periodic grid on [-1,1), IC E(x,0)=sin(pi x), exact E(x,t)=sin(pi x - c*pi t).
    Compares quantum update to classical explicit leapfrog.
    """
    sim = provide_simulator(noise=noise)

    # periodic grid
    x = np.linspace(-1.0, 1.0, N, endpoint=False)
    dx = 2.0 / N
    dt = CFL * dx / c
    lam = CFL  # = c*dt/dx

    print(f"\nWave test: N={N}, steps={steps}, dx={dx:.6g}, dt={dt:.6g}, c={c}, CFL={lam:.3f}")

    E0 = np.sin(np.pi * x)
    E1 = wave_first_step_taylor(E0, dx, dt, c)
    E_prev_q,  E_curr_q  = E0.copy(), E1.copy()
    get_circuit_info(E_prev_q, E_curr_q, lam, batch_size=batch_size, sim=sim)

    profiles = {}
    L_inf_tot = []
    L2_tot = []
    for ss in shots_list:
        L2 = []
        L_inf = []
        for r in range(repetitions):
            # IC
            E0 = np.sin(np.pi * x)
            E1 = wave_first_step_taylor(E0, dx, dt, c)

            # copies for classical and quantum feeds
            E_prev_ref, E_curr_ref = E0.copy(), E1.copy()
            E_prev_q,  E_curr_q  = E0.copy(), E1.copy()
            print(f"Wave: shots={ss}, rep={r+1}/{repetitions}")
            for n in range(1, steps):
                print("Step ", n, "/",steps)
                # classical
                E_next_ref = wave_eq_one_step_classical(E_prev_ref, E_curr_ref, lam)
                # quantum
                E_next_q = one_step_wave_eq_batched(E_prev_q, E_curr_q, lam, shots=ss, batch_size=batch_size, sim=sim)

                E_prev_ref, E_curr_ref = E_curr_ref, E_next_ref
                E_prev_q,  E_curr_q  = E_curr_q,  E_next_q

            # exact at T = steps*dt
            T = steps * dt
            E_exact = np.sin(np.pi * x - c * np.pi * T)

            diff = E_curr_q - E_exact
            linf = np.max(np.abs(diff))
            l2 = np.sqrt(np.mean(diff**2))
            L_inf.append(linf)
            L2.append(l2)

            if r == 0:
                profiles[ss] = (x.copy(), E_curr_ref.copy(), E_curr_q.copy(), E_exact.copy(), T)

        L_inf_tot.append(L_inf)
        L2_tot.append(L2)
        print(f"{ss:<9d} {np.mean(L_inf):.4f} ± {np.std(L_inf):.4f}  {np.mean(L2):.4f} ± {np.std(L2):.4f}")
    
    print("\nWave eq - errors")
    print("\nshots M    L_inf (mean ± std)    L2 (mean ± std)")
    for i, ss in enumerate(shots_list):
        L_inf = L_inf_tot[i]
        L2 = L2_tot[i]
        print(f"{ss:<9d} {np.mean(L_inf):.4f} ± {np.std(L_inf):.4f}  {np.mean(L2):.4f} ± {np.std(L2):.4f}")
 
    # Plot one representative profile per shots choice
    fig = plt.figure()
    xx, E_ref, E_q, E_ex, T = profiles[shots_list[0]]
    plt.plot(xx, E_ex, label=f"Analytic (T={T:.3f})")
    for ss in shots_list:
        xx, E_ref, E_q, E_ex, T = profiles[ss]
        #plt.plot(xx, E_ref, "--", label=f"classical")
        plt.plot(xx, E_q, linestyle='--', marker="x", label=f"Quantum (M={ss})")

    plt.xlabel("x"); plt.ylabel("E(x,T)")
    plt.title("Wave: classical vs quantum")
    plt.grid(True); plt.legend(); plt.tight_layout()
    name_fig = f"wave_steps{steps}_N{N}_CFL{CFL:.2f}_batch{batch_size}_noise{noise}.pdf"
    fig.savefig(name_fig, dpi=600, bbox_inches="tight")
    plt.show(); plt.close()

if __name__ == "__main__":
    print("Start wave experiments")
    wave_test(N=N_GRID_POINTS, steps=STEPS, shots_list=SHOT_LIST,
              repetitions=REPETITIONS, batch_size=BATCH_SIZE, noise=NOISE)
