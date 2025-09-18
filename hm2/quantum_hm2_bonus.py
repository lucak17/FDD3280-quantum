import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2

print(qiskit.__version__, qiskit_ibm_runtime.__version__, qiskit_aer.__version__)

DTYPE = np.complex128  

NQUBIT = 3


def basis_state(bits):
    dim = 2**NQUBIT
    index = sum(b * (2**i) for i, b in enumerate(bits)) 
    state = np.zeros(dim)
    state[index] = 1.0
    return state


I = np.eye(2, dtype=DTYPE)
H = (1/np.sqrt(2)) * np.array([[1,  1],
                               [1, -1]], dtype=DTYPE)

def one_qubit_op_on_target_matrix_lsb_first(n_qubits: int, gate: np.ndarray, target: int) -> np.ndarray:
    """
    Build the 2**n x 2**n matrix applying 1-qubit `gate` to `target`,
    using LITTLE-ENDIAN indexing (q0 is LSB) and looping from LSB->MSB.

    We left-kron the current factor so the final order is (MSB ... LSB).
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if not (0 <= target < n_qubits):
        raise ValueError(f"target must be in [0, {n_qubits-1}]")
    if gate.shape != (2, 2):
        raise ValueError("gate must be 2x2")

    U = np.array([[1.0]], dtype=DTYPE)
    for q in range(0, n_qubits):                 # LSB (q=0) -> MSB (q=n-1)
        factor = gate if q == target else I
        U = np.kron(factor, U)                   # left-kron to keep MSB...LSB order
    return U

def hadamard_on_qubit_matrix(n_qubits: int, target: int) -> np.ndarray:
    return one_qubit_op_on_target_matrix_lsb_first(n_qubits, H, target)



def initialize_register(qc, qr, init_value):

    for idx, v in enumerate(init_value):
        if v==1:
            qc.x(qr[idx])
    return


def swapt_test_N_qubits_circuit(reference, test):

    reference_register = QuantumRegister(NQUBIT)
    test_register = QuantumRegister(NQUBIT)
    ancilla = QuantumRegister(1)
    output = ClassicalRegister(1)

    qc = QuantumCircuit(reference_register, test_register, ancilla, output)

    initialize_register(qc, reference_register, reference)
    initialize_register(qc, test_register, test)


    qc.h(ancilla[0])
    for i in range(NQUBIT):
        qc.cswap(ancilla[0], reference_register[i], test_register[i])
    qc.h(ancilla[0])
    qc.measure(ancilla[0], output[0])
    return qc


def make_and_run_swap(reference, test, shots):

    sim_sampler =AerSimulator()    
    circuit = swapt_test_N_qubits_circuit(reference, test)
    fig1 = circuit.draw(output="mpl")
    fig1.savefig("swap_circuit.png", dpi=400, bbox_inches="tight")
    # Simulator
    sim_counts = sim_sampler.run([circuit], shots=shots).result().get_counts()
    
    #print(U3_q2)
    F_exact = float(np.abs(np.vdot(basis_state(reference), basis_state(test)))**2)
    p1 = sim_counts.get('1', 0)/shots
    print("Reference (q0,q1,q2): ", reference, " Test (q0,q1,q2): ", test)
    print("Counts:", sim_counts)
    print("P(output=1): ",  p1, 
          " - F=1-2P(1)=",1-2* p1, "+-", 2*np.sqrt(p1*(1-p1)/shots),
          " - F_exact=", F_exact,
          "\n")


## Task 1
SHOTS = 2000

# initial state for register R, with little endian notation -> |5>
init_R = [1, 0, 1]

# initial state for register L, with little endian notation
init_L = [1, 0, 1] 
make_and_run_swap(init_R,init_L,shots=SHOTS)


# initial state for register L, with little endian notation
# random state
rng = np.random.default_rng()
init_L = [rng.integers(0, 2), rng.integers(0, 2), rng.integers(0, 2)] 
make_and_run_swap(init_R,init_L,shots=SHOTS)

