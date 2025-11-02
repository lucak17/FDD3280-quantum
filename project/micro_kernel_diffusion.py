import os
import numpy as np, matplotlib.pyplot as plt

BCS = 0.0
N_GRID_POINTS = 64
LAMBDA = 0.25
NU_DIFFUSION = 1.0
STEPS = 60
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

def build_batch(u: np.ndarray, lam: float, start_idx: int, batch_size: int, name="batch"):
    """
    Make one circuit containing up to M kernels for nodes i = start_idx .. start_idx+M-1 (clipped to N).
    Returns (circuit, node_indices) where node_indices are the interior node ids covered.
    """
    n_points = u.size - 2
    wL, wC, wR = lam, 1.0 - 2.0 * lam, lam

    # nodes that go into this batch
    nodes = list(range(start_idx, min(start_idx + batch_size, n_points + 1)))
    mm = len(nodes)

    q = QuantumRegister(3 * mm, "q")
    c = ClassicalRegister(mm, "c")
    circ = QuantumCircuit(q, c, name=name)

    for j, i in enumerate(nodes):
        s0 = 3 * j
        s1 = s0 + 1
        ro = s0 + 2
        inst = branching_mk_instruction(u[i - 1], u[i], u[i + 1], wL, wC, wR, name=f"node_{i}")
        circ.append(inst, [q[s0], q[s1], q[ro]])
    ro_qubits = [q[3*j + 2] for j in range(mm)]
    circ.measure(ro_qubits, c[:mm])
    return circ, nodes

def one_step_diffusion_eq_batched(u: np.ndarray, lam: float, shots: int, batch_size: int, sim=None) -> np.ndarray:
    if sim is None:
        sim = provide_simulator()

    n_points = u.size - 2
    u_new = np.ones_like(u) * BCS

    # build all batch circuits first
    batches = []
    idx_groups = []
    for start in range(1, n_points + 1, batch_size):
        bc, nodes = build_batch(u, lam, start, batch_size, name=f"batch_{start}")
        batches.append(bc)
        idx_groups.append(nodes)

    # transpile & run as a list
    tbatches = transpile(batches, backend=sim, optimization_level=3, seed_transpiler=42)
    result = sim.run(tbatches, shots=shots).result()

    # P(bit j == '1') from raw counts of an m-bit circuit, invert endianess (Qiskit is little-endian)
    # Usefull is batch_size>1 is used
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
            u_new[i] = p1_from_counts(counts, m, j)

    return u_new


def get_circuit_info(u: np.ndarray, lam: float, batch_size: int, sim=None):
    if sim is None:
        sim = provide_simulator()

    n_points = u.size - 2
    # build all batch circuits first
    batches = []
    idx_groups = []
    for start in range(1, n_points + 1, batch_size):
        bc, nodes = build_batch(u, lam, start, batch_size, name=f"batch_{start}")
        batches.append(bc)
        idx_groups.append(nodes)

    # transpile
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

# ---------------------------
# Experiments
# ---------------------------
def convergence_test(N=65, lam=0.25, steps=47, shots_list=(1000,4000,8000,16000,32000), repetitions=10, nu=1.0,
                     one_step_circuit=one_step_diffusion_eq_batched, noise=False):
    
    sim = provide_simulator(noise=noise)
    dx = 1.0 / (N + 1)
    dt = lam * dx * dx / nu
    T = steps * dt

    x = np.linspace(0, 1, N+2)
    u_analytic = np.zeros(N+2) + BCS
    u_analytic[1:-1] = np.exp(-nu * (np.pi**2) * T) * np.sin(np.pi * x[1:-1]) + BCS
    
    print(f"\nConvergence test: N={N}, steps={steps}, λ={lam}, ν={nu}, dx={dx:.6g}, dt={dt:.6g}, T={T:.6g}")
    
    # print qc metrics
    u = np.zeros(N+2) + BCS
    u[1:-1] = np.sin(np.pi * x[1:-1]) + BCS
    get_circuit_info(u, lam, batch_size=BATCH_SIZE, sim=sim)

    profiles_for_plot = {}
    L_inf_tot = []
    L2_tot = []
    for ss in shots_list:
        L_inf = []
        L2 = []
        for r in range(repetitions):
            # Initial condition u(x,0)=sin(pi x)
            u = np.zeros(N+2) + BCS
            u[1:-1] = np.sin(np.pi * x[1:-1]) + BCS
            print(f"Diffusion: shots={ss}, rep={r+1}/{repetitions}")
            for i in range(steps):
                print("Step ", i+1, "/",steps)
                u = one_step_circuit(u, lam, shots=ss, batch_size=BATCH_SIZE, sim=sim)

            # errors
            diff = u[1:-1] - u_analytic[1:-1]
            linf = np.max(np.abs(diff))
            l2 = np.sqrt(np.mean(diff**2))
            L_inf.append(linf)
            L2.append(l2)

            if r == 0:
                profiles_for_plot[ss] = u.copy()
        L_inf_tot.append(L_inf)
        L2_tot.append(L2)
        print(f"{ss:<9d} {np.mean(L_inf):.4f} ± {np.std(L_inf):.4f}  {np.mean(L2):.4f} ± {np.std(L2):.4f}")

    print("\nDiffusion eq - errors")
    print("\nshots M    L_inf (mean ± std)    L2 (mean ± std)")
    for i, ss in enumerate(shots_list):
        L_inf = L_inf_tot[i]
        L2 = L2_tot[i]
        print(f"{ss:<9d} {np.mean(L_inf):.4f} ± {np.std(L_inf):.4f}  {np.mean(L2):.4f} ± {np.std(L2):.4f}")
    
    # Plot analytic vs quantum
    fig1 = plt.figure()
    plt.plot(x[:], u_analytic[:], label="Analytic")
    for ss in shots_list:
        plt.plot(x[:], profiles_for_plot[ss][:], linestyle='--', marker="x", label=f"Quantum (M={ss})")
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title(f"Heat equation Classic vs Quantum (T={T:.4g})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    name_fig = f"diffusion_steps{steps}_noise{noise}.pdf"
    fig1.savefig(name_fig, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    
    fig2 = plt.figure()
    for ss in shots_list:
        plt.plot(x[:], u_analytic[:] - profiles_for_plot[ss][:], linestyle='--', marker="x", label=f"Quantum (M={ss})")
    plt.xlabel("x")
    plt.ylabel("u_c(x, T) - u_q(x, T)")
    plt.title(f"Heat equation Classic vs Quantum - Error (T={T:.4g})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    name_fig = f"diffusion_steps{steps}_batch{BATCH_SIZE}_noise{noise}_error.pdf"
    fig2.savefig(name_fig, dpi=600, bbox_inches="tight")
    plt.show()

# ---------------------------
# Run the experiments
# ---------------------------
if __name__ == "__main__":

    print("Start exepriments")
    #Convergence, print L_inf/L2 stats
    convergence_test(N=N_GRID_POINTS, lam=LAMBDA, steps=STEPS,
                    shots_list=SHOT_LIST, repetitions=REPETITIONS, nu=NU_DIFFUSION, 
                    one_step_circuit=one_step_diffusion_eq_batched, noise=NOISE)