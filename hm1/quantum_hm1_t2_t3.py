import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator


# k‑bit QRNG
def qrng(k: int):
    qc = QuantumCircuit(k)
    for q in range(k):
        qc.h(q)          # one coin‑flip per qubit
    qc.measure_all()
    return qc


print(qiskit.__version__, qiskit_ibm_runtime.__version__, qiskit_aer.__version__)

service = QiskitRuntimeService()


## Task 2

cands = service.backends(simulator=False, operational=True, min_num_qubits=6)
for b in cands: print(b.name, b.num_qubits)

A = service.least_busy(simulator=False, operational=True, min_num_qubits=6)
B = next(b for b in cands if b.name != A.name)
print("Resource A: ",A)
print("Resource B: ",B)


k = 6
qc = qrng(k)
qc.draw()



## Task 3

pmA = generate_preset_pass_manager(optimization_level=3, backend=A)
isaA = pmA.run(qc)

pmB = generate_preset_pass_manager(optimization_level=3, backend=B)
isaB = pmB.run(qc)

print("A ops:", isaA.count_ops(), "depth:", isaA.depth())
print("B ops:", isaB.count_ops(), "depth:", isaB.depth())

# (Optional) See which physical qubits were chosen for your logical qubits 0..k-1
print("ISA A initial_index_layout:", isaA.layout.initial_index_layout())
print("ISA A routing_permutation: ", isaA.layout.routing_permutation())
print("ISA A final_index_layout:  ", isaA.layout.final_index_layout())
print("ISA B final_index_layout:  ", isaB.layout.final_index_layout())

# (Optional) Peek at the device’s native gate names (you don't need to know them yet)
print("A basis gates:", A.configuration().basis_gates)
print("B basis gates:", B.configuration().basis_gates)
# Draw the transpiled circuit
isaA.draw()


# Hardware: target a specific backend
sampler = SamplerV2(mode=A) # or mode=B
result = sampler.run([isaA], shots=1000).result()
counts = result[0].data.meas.get_counts() # {'010011': n, ...}
total = sum(counts.values())
probs = {bitstr: count / total for bitstr, count in counts.items()}

# Simulator with the same result schema
sim_counts = BackendSamplerV2(backend=AerSimulator()).run([isaA], shots=4000).result()[0].data.meas.get_counts()
print("Counts:", sim_counts)