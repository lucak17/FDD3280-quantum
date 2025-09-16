import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime, qiskit_aer
import math
import matplotlib.pyplot as plt


DTYPE = np.complex128


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

    def viz_circle(self, max_cols: int = 8, figsize_scale: float = 2.3):
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
        plt.tight_layout()
        plt.show()


def ket00():
    """Return |00> as a length‑4 complex column vector."""
    v = np.zeros(4, dtype=DTYPE)
    v[0] = 1.0
    return v

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return v / n


# 1‑qubit and 2‑qubit gate definitions (matrix form)
I2 = np.array([[1, 0], [0, 1]], dtype=DTYPE)
H  = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# CNOT with control = q0 (MSB), target = q1 (LSB) in basis |q0 q1>
CNOT_01 = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
], dtype=DTYPE)


# CNOT with control = q1 (LSB), target = q0 (MSB) in basis |q0 q1>
CNOT_10 = np.array([
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,1,0,0]
], dtype=DTYPE)

# Expand a 1‑qubit unitary U to a 2‑qubit operator acting on target qubit t in |q0 q1> order
# t = 0 applies to q0 (MSB); t = 1 applies to q1 (LSB)
def expand_1q(U: np.ndarray, t: int) -> np.ndarray:
    if t == 0:   # act on q0 (MSB)
        return np.kron(U, I2)
    elif t == 1: # act on q1 (LSB)
        return np.kron(I2, U)
    else:
        raise ValueError("target index must be 0 or 1")

# Apply a gate matrix to a state vector
def apply(U: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return U @ psi


def bell_state() -> np.ndarray:
    psi = ket00()
    # H on q0 (MSB), then CNOT(control=q0, target=q1)
    print("Initial state: ", np.round(psi, 6))
    psi = apply(expand_1q(H, t=0), psi)
    psi = apply(CNOT_01, psi)
    return normalize(psi)


## Task 1

print("qiskit version: ",qiskit.__version__, 
      "\nqiskit_ibm_runtime version: ", qiskit_ibm_runtime.__version__, 
      "\nqiskit_aer version: ", qiskit_aer.__version__)


psi = ket00()
assert np.isclose(np.linalg.norm(psi), 1.0)
print(psi)

# Quick unit tests
assert np.allclose(H.conj().T @ H, I2)
assert np.allclose(CNOT_01.conj().T @ CNOT_01, np.eye(4))
assert np.allclose(CNOT_10.conj().T @ CNOT_10, np.eye(4))


psi_bell = bell_state()
print("Bell state vector:", np.round(psi_bell, 6))


BITSTR = ["00","01","10","11"]

def sample_bitstrings(psi: np.ndarray, shots: int = 10_000, seed: int = 7):
    rng = np.random.default_rng(seed)
    probs = np.abs(psi)**2
    idx = rng.choice(4, size=shots, p=probs)
    return idx

# Sample from Bell state
shots = 10_000
idx = sample_bitstrings(psi_bell, shots=shots)

# Make histogram
counts = {b: int(np.sum(idx == i)) for i, b in enumerate(BITSTR)}
print(counts)
fig1 = plt.figure()
plt.bar(counts.keys(), counts.values())
plt.xlabel("bitstring"); plt.ylabel("counts"); plt.title("Bell state measurement histogram")
fig1.savefig("bell_state_hist.png", dpi=400)
plt.show()

QubitSystem(psi_bell, label="Bell state (|00⟩+|11⟩)/√2").viz_circle()



## Task 2

def expand_1q_n(U: np.ndarray, t: int, n: int) -> np.ndarray:
    op = np.array([[1]], dtype=U.dtype)
    for q in range(n):
        op = np.kron(op, U if q == t else I2)
    return op

# Apply CSWAP, with ancilla as control
P0 = np.array([[1,0],[0,0]], dtype=DTYPE)  # |0><0| on ancilla
P1 = np.array([[0,0],[0,1]], dtype=DTYPE)  # |1><1| on ancilla
SWAP_A0_B0 = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]
], dtype=DTYPE)
CSWAP = np.kron(P0, np.eye(4, dtype=DTYPE)) + np.kron(P1, SWAP_A0_B0)

def swap_test_fidelity(psi: np.ndarray, phi: np.ndarray, verbose: bool = True):
    # Ensure proper shapes and normalization
    psi = normalize(np.asarray(psi, dtype=DTYPE).flatten())
    phi = normalize(np.asarray(phi, dtype=DTYPE).flatten())

    # |0>_output ⊗ |ψ>_A ⊗ |φ>_B
    init = np.kron(np.array([1,0], dtype=DTYPE), np.kron(psi, phi))

    # H on output (MSB), CSWAP(A,B) controlled by output, then H on output
    U_Ho = expand_1q_n(H, t=0, n=3)
    state = U_Ho @ init
    state = CSWAP @ state
    state = U_Ho @ state

    # Fidelity
    prob = np.abs(state)**2
    assert np.isclose(prob.sum(), 1.0), "Total probability must be 1."
    prob1 = float(prob[4:].sum())  # indices with output=1 (MSB=1 → indices 4..7)
    F_hat = 1.0 - 2.0*prob1

    # Exact fidelity from inner product
    F_exact = float(np.abs(np.vdot(psi, phi))**2)

    return F_hat, F_exact, prob1, state


# Test states
ket0 = np.array([1,0], dtype=DTYPE)
ket1 = np.array([0,1], dtype=DTYPE)
ket_plus = normalize(H @ ket0)

tests = [
    ("psi = phi = 0 ", ket0, ket0),
    ("psi = 0, phi = 1 ", ket0, ket1),
    ("psi = +, phi = 0 ", ket_plus, ket0),
]

print("Fidelity test:")
for name, psi, phi in tests:
    print(f"-- {name} --")
    F_hat, F_exact, prob1, state = swap_test_fidelity(psi, phi, verbose=True)
    # Sanity checks vs expected analytic values
    print("F_exact: ", F_exact, " F_hat: ", F_hat)
    print("P(output=1): ", prob1)

# Random example
rng = np.random.default_rng(534)
psi_rand = normalize(rng.normal(size=2) + 1j*rng.normal(size=2))
phi_rand = normalize(rng.normal(size=2) + 1j*rng.normal(size=2))
print("\n-- Random states example --")
F_hat_r, F_exact_r, P1_r, _ = swap_test_fidelity(psi_rand, phi_rand, verbose=True)
print("F_exact: ", F_exact_r, " F_hat: ", F_hat_r)
print("P(output=1): ", P1_r)
