import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2

print(qiskit.__version__, qiskit_ibm_runtime.__version__, qiskit_aer.__version__)

DTYPE = np.complex128  

NQUBIT = 3


def initialize_register(qc, qr, init_value):

    for idx, v in enumerate(init_value):
        if v==1:
            qc.x(qr[idx])
    return


def swapt_test_N_qubits(reference, test):

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


def make_and_run_swap(reference, test):

    sim_sampler =AerSimulator()    
    circuit = swapt_test_N_qubits(reference, test)
    fig1 = circuit.draw(output="mpl")
    #fig1.savefig("swap_circuit.png", dpi=400, bbox_inches="tight")
    # Simulator
    sim_counts = sim_sampler.run([circuit], shots=2000).result().get_counts()
    print("Reference (q0,q1,q2): ", reference, " Test (q0,q1,q2): ", test)
    print("Counts:", sim_counts)


## Task 1

# initial state for register R, with little endian notation -> |5>
init_R = [1, 0, 1]

# initial state for register L, with little endian notation
init_L = [1, 0, 1] 
make_and_run_swap(init_R,init_L)


# initial state for register L, with little endian notation
init_L = [1, 0, 0] 
make_and_run_swap(init_R,init_L)

