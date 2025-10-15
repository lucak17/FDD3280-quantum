import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import math
import matplotlib.pyplot as plt

print(qiskit.__version__, qiskit_ibm_runtime.__version__, qiskit_aer.__version__)

DTYPE = np.complex128

def sv_from_circ(circ: QuantumCircuit) -> np.ndarray:
    sv = Statevector(circ)
    return np.asarray(sv, dtype=np.complex128)

class QubitSystem:
    def __init__(self, statevector: np.ndarray, label: str = "Qubit System"):
        self.label = label
        self.set_statevector(statevector)

    def set_statevector(self, statevector: np.ndarray):
        sv = np.asarray(statevector, dtype=np.complex128).flatten()
        if sv.ndim != 1:
            raise ValueError("Statevector must be 1D.")
        n_states = sv.size
        n_qubits = int(round(math.log2(n_states)))
        if 2**n_qubits != n_states:
            raise ValueError("Length must be a power of 2.")
        # Defensive normalization (harmless if already normalized)
        norm = np.linalg.norm(sv)
        if norm != 0 and not np.isclose(norm, 1.0):
            sv = sv / norm

        self.n_qubits = n_qubits
        self.n_states = n_states
        self.amps  = sv
        self.prob  = np.abs(sv)**2
        self.phase = np.angle(sv)

    def viz_circle(self, max_cols: int = 8, figsize_scale: float = 2.3, name_fig=None):
        cols = max(1, min(max_cols, self.n_states))
        rows = int(math.ceil(self.n_states / cols))

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols*figsize_scale, rows*(figsize_scale+0.2))
        )
        axes = np.atleast_2d(axes)

        def bitstr(i: int, n: int) -> str:
            return format(i, f"0{n}b")

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.set_aspect("equal")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")

            if idx >= self.n_states:
                ax.set_visible(False)
                continue

            # Outer reference circle
            ax.add_patch(plt.Circle((0.5, 0.5), 0.48, fill=False, lw=1.0, alpha=0.5))

            # Filled disk: radius ∝ sqrt(probability) so area ∝ probability
            radius = 0.48 * np.sqrt(self.prob[idx])
            ax.add_patch(plt.Circle((0.5, 0.5), radius, alpha=0.25))

            # Phase arrow
            angle = self.phase[idx]
            L = 0.45
            x2 = 0.5 + L * np.cos(angle)
            y2 = 0.5 + L * np.sin(angle)
            ax.arrow(0.5, 0.5, x2 - 0.5, y2 - 0.5,
                     head_width=0.03, head_length=0.05, length_includes_head=True)

            ax.set_title(f"|{bitstr(idx, self.n_qubits)}⟩", fontsize=10)

        fig.suptitle(self.label, fontsize=12)
        if name_fig:
            fig.savefig(name_fig, dpi=400, bbox_inches="tight")
        plt.tight_layout()
        plt.show()



def add1_circuit(qc, q_reg):
    n = len(q_reg)
    for k in range(n-1):
        controls = [q_reg[i] for i in range(n - 1 - k)]
        target = q_reg[n -1 -k]
        qc.mcx(controls, target)
    qc.x(q_reg[0])
    return qc

def subtract1_circuit(qc, q_reg):
    n = len(q_reg)
    qc.x(q_reg[0])
    for k in range(1, n):
        controls = [q_reg[i] for i in range(k)]
        target = q_reg[k]
        qc.mcx(controls, target)
    return qc


def control_add1_circuit(qc, q_reg, q_control):
    n = len(q_reg)
    for k in range(n-1):
        controls = list(q_control) + [q_reg[i] for i in range(n - 1 - k)]
        target = q_reg[n -1 -k]
        qc.mcx(controls, target)
    qc.mcx(list(q_control), q_reg[0])
    return qc

def control_subtract1_circuit(qc, q_reg, q_control):
    n = len(q_reg)
    qc.mcx(list(q_control), q_reg[0])
    for k in range(1, n):
        controls = list(q_control) + [q_reg[i] for i in range(k)]
        target = q_reg[k]
        qc.mcx(controls, target)
    return qc


def initialize_register(qc, q_reg):
    n = len(q_reg)
    qc.x(q_reg[n-1])
    #qc.h(q_reg[n-3])
    return qc

def random_walk_step(qc, q_reg, q_coin):

    qc.h(q_coin[0])
    control_add1_circuit(qc,q_reg,q_coin)
    qc.x(q_coin[0])
    control_subtract1_circuit(qc,q_reg,q_coin)
    qc.x(q_coin[0])
    return qc

def random_walk_circuit(nqubit, nstep):

    q_reg = QuantumRegister(nqubit)
    q_coin = QuantumRegister(1)
    qc = QuantumCircuit(q_reg, q_coin)
    qc = initialize_register(qc,q_reg)
    qc.x(q_coin[0])
    for i in range(nstep):
        qc = random_walk_step(qc, q_reg, q_coin)    

    return qc, q_reg, q_coin


def build_measured_circuit(nqubit, nstep, init_middle=True):
    qc, q_reg, q_coin = random_walk_circuit(nqubit, nstep)
    # measure ONLY the position register (ignore coin for distribution)
    c_reg = ClassicalRegister(nqubit, "c_pos")
    qc.add_register(c_reg)
    qc.measure(q_reg, c_reg)
    return qc


def random_walk_step_roty(qc, q_reg, q_coin, theta):
    # Coin: RY(theta)
    qc.ry(theta, q_coin[0])

    # Shift right if coin==|1>
    control_add1_circuit(qc, q_reg, q_coin)

    # Shift left if coin==|0>  (X sandwich)
    qc.x(q_coin[0])
    control_subtract1_circuit(qc, q_reg, q_coin)
    qc.x(q_coin[0])
    return qc

# --- build a circuit with RY coin for nstep steps ---
def random_walk_circuit_roty(nqubit, nstep, theta):
    q_reg = QuantumRegister(nqubit)
    q_coin = QuantumRegister(1)
    qc = QuantumCircuit(q_reg, q_coin)

    # same position init as your code
    qc = initialize_register(qc, q_reg)
    #qc.x(q_coin[0])
    for _ in range(nstep):
        qc = random_walk_step_roty(qc, q_reg, q_coin, theta)

    return qc, q_reg, q_coin

def build_measured_circuit_roty(nqubit, nstep, theta):
    qc, q_reg, q_coin = random_walk_circuit_roty(nqubit, nstep, theta)
    # measure ONLY the position register
    c_reg = ClassicalRegister(nqubit, "c_pos")
    qc.add_register(c_reg)
    qc.measure(q_reg, c_reg)
    return qc


## Task 1

NQUBIT = 7
NSTEP = 30
NSHOTS = 10000

#qc = build_measured_circuit(NQUBIT, NSTEP)


theta = np.pi * 0.25
qc = build_measured_circuit_roty(NQUBIT, NSTEP, theta)

#QubitSystem(sv_from_circ(qc), label="Initial State").viz_circle()
#fig1 = qc.draw(output="mpl")
#name_fig = f"q_circuit_random_walk_{NSTEP}steps.png"
#fig1.savefig(name_fig, dpi=400, bbox_inches="tight")

sim_sampler =AerSimulator()    
sim_counts = sim_sampler.run([qc], shots=NSHOTS).result().get_counts()

shots = sum(sim_counts.values())
num_states = 2 ** NQUBIT
xs = np.arange(num_states)
#print(xs)
ps = np.array([
    sim_counts.get(format(i, f'0{NQUBIT}b'), 0) / shots
    for i in xs
])
order = np.argsort(xs)
xs, ps = xs[order], ps[order]

fig1 = plt.figure(figsize=(10, 3))
plt.bar(xs, ps, width=1.0)
plt.xlabel("Position")
plt.ylabel("Probability")
plt.title(f"Position distribution after {NSTEP} steps")
plt.tight_layout()
name_fig = f"q_random_walk_{NSTEP}steps.png"
fig1.savefig(name_fig, dpi=400, bbox_inches="tight")
plt.show()