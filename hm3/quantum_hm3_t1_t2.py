import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
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

    qc.ccx(q_reg[0], q_reg[1], q_reg[2])  # propagate carry to bit 2 (Toffoli)
    qc.cx(q_reg[0], q_reg[1])        # propagate carry to bit 1
    qc.x(q_reg[0])                   # flip LSB
    return qc

def subtract1_circuit(qc, q_reg):

    qc.x(q_reg[0])                   # flip LSB
    qc.cx(q_reg[0], q_reg[1])        # propagate carry to bit 1
    qc.ccx(q_reg[0], q_reg[1], q_reg[2])  # propagate carry to bit 2 (Toffoli)
    
    return qc



def prepare_state_qft(n_qubit):
    q_reg = QuantumRegister(n_qubit)    
    qc = QuantumCircuit(q_reg)
    
    for i in range(n_qubit):
        qc.h(q_reg[i])
    qc.p(np.pi, q_reg[1])

    return qc, q_reg

def qft_circuit(qc, q_reg, n_qubit):

    idx_last = n_qubit
    for i in range(n_qubit-1):
        idx_last = idx_last - 1
        qc.h(q_reg[idx_last])
        phase = -0.5 * np.pi
        for j in range(idx_last):
            qc.cp(phase, q_reg[idx_last - 1 - j], q_reg[idx_last])
            phase = phase/2
    qc.h(q_reg[0])
    for i in range( n_qubit // 2 ):
        qc.swap(q_reg[i], q_reg[n_qubit - 1 - i])

    return qc

## Task 1

NQUBIT_TASK1 = 3

general = False
overflow = False
underflow = False

if general:
    q_reg = QuantumRegister(NQUBIT_TASK1)    
    qc = QuantumCircuit(q_reg)
    qc.h(q_reg[1])
    QubitSystem(sv_from_circ(qc), label="Initial State").viz_circle()
    qc = add1_circuit(qc, q_reg)
    QubitSystem(sv_from_circ(qc), label="Increment: +1").viz_circle()
    qc = subtract1_circuit(qc, q_reg)
    QubitSystem(sv_from_circ(qc), label="Decrement: -1").viz_circle()


# overflow
if overflow:
    q_reg = QuantumRegister(NQUBIT_TASK1)    
    qc = QuantumCircuit(q_reg)
    qc.x(q_reg[0])
    qc.x(q_reg[1])
    qc.x(q_reg[2])
    qc.h(q_reg[0])
    QubitSystem(sv_from_circ(qc), label="Initial State").viz_circle()
    qc = add1_circuit(qc, q_reg)
    QubitSystem(sv_from_circ(qc), label="Increment: +1").viz_circle()

# underflow
if underflow:
    q_reg = QuantumRegister(NQUBIT_TASK1)    
    qc = QuantumCircuit(q_reg)
    qc.h(q_reg[0])
    QubitSystem(sv_from_circ(qc), label="Initial State").viz_circle()
    qc = subtract1_circuit(qc, q_reg)
    QubitSystem(sv_from_circ(qc), label="Decrement: -1").viz_circle()



# Task 2
NQUBIT_TASK2 = 4
qc, q_reg = prepare_state_qft(NQUBIT_TASK2)
QubitSystem(sv_from_circ(qc), label="Initial State").viz_circle(name_fig="qft_initial_state.png")

qc = qft_circuit(qc, q_reg, NQUBIT_TASK2)
QubitSystem(sv_from_circ(qc), label="QFT").viz_circle(name_fig="qft_after_qft.png")

fig1 = qc.draw(output="mpl")
name_fig = "qft_circuit.png" 
fig1.savefig(name_fig, dpi=400, bbox_inches="tight")
