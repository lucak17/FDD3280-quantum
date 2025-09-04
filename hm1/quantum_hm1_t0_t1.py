import os
import statistics
import matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_coupling_map


def print_backend_config(backend):
    prop = backend.properties()
    config = backend.configuration()
    nqubit = config.n_qubits
    qubits = range(nqubit)
    basis_gates = config.basis_gates
    
    # qubits properties
    t1s  = [prop.t1(q) for q in qubits] # seconds
    t2s  = [prop.t2(q) for q in qubits] # seconds
    rOutErr = [prop.readout_error(q) for q in qubits] # prob
    rOutLen = [prop.readout_length(q) for q in qubits] # seconds

    # queue 
    queue = backend.status()
    print("\nDevice configuration:")
    print("backend: ", backend)
    print("num_qubits: ", nqubit)
    print("basis_gates: ", basis_gates)
    print("queue_jobs: ", queue.pending_jobs)
    print("Median qubits properties: ")
    print("T1_us: ", statistics.median(t1s) * 1e6)
    print("T2_us: ", statistics.median(t2s) * 1e6)
    print("readout_error: ", statistics.median(rOutErr))
    print("readout_length_us: ", statistics.median(rOutLen) * 1e6)
    print("\n\n")

# Task 0
print("qiskit version: ",qiskit.__version__, 
      "\nqiskit_ibm_runtime version: ", qiskit_ibm_runtime.__version__, 
      "\nqiskit_aer version: ", qiskit_aer.__version__)


# Task 1
# Read environmental variables
TOKEN = os.environ.get("TOKEN_QISKIT", None)  # REQUIRED
INSTANCE = os.environ.get("INSTANCE_QISKIT", None)  # OPTIONAL: e.g., "crn:v1:bluemix:public:quantum-computing:us-east:...:..."

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=TOKEN.strip(),
    instance=(INSTANCE.strip() if isinstance(INSTANCE, str) and INSTANCE.strip() else None),
    set_as_default=True,
    overwrite=True,
)
print("Saved default account for this runtime.")

service = QiskitRuntimeService()

# get resources
cands = service.backends(simulator=False, operational=True, min_num_qubits=6)
for b in cands: print(b.name, b.num_qubits)
A = service.least_busy(simulator=False, operational=True, min_num_qubits=6)
B = next(b for b in cands if b.name != A.name)

# print resource properties
print_backend_config(A)
print_backend_config(B)

cmapA = A.coupling_map
cmapB = B.coupling_map
figA = plot_coupling_map(A.num_qubits, None, cmapA.get_edges())
figB = plot_coupling_map(B.num_qubits, None, cmapB.get_edges())
figA.savefig("coupling_A.png", dpi=400, bbox_inches="tight")
figB.savefig("coupling_B.png", dpi=400, bbox_inches="tight")

#plt.show()