import os
import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_coupling_map


# Task 0
print("qiskit version: ",qiskit.__version__, 
      "\nqiskit_ibm_runtime version: ", qiskit_ibm_runtime.__version__, 
      "\nqiskit_aer version: ", qiskit_aer.__version__)


# Task 1
# Read environmental variables
TOKEN = os.environ.get("TOKEN_QISKIT")  # REQUIRED
INSTANCE = None  # OPTIONAL: e.g., "crn:v1:bluemix:public:quantum-computing:us-east:...:..."

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=TOKEN.strip(),
    instance=(INSTANCE.strip() if isinstance(INSTANCE, str) and INSTANCE.strip() else None),
    set_as_default=True,
    overwrite=True,
)
print("Saved default account for this runtime.")

service = QiskitRuntimeService()

cands = service.backends(simulator=False, operational=True, min_num_qubits=6)
for b in cands: print(b.name, b.num_qubits)

A = service.least_busy(simulator=False, operational=True, min_num_qubits=6)
B = next(b for b in cands if b.name != A.name)
print("Resource A: ",A)
print("Resource B: ",B)


cfgA = A.configuration() 
cfgB = B.configuration()
print("A basis_gates:", cfgA.basis_gates)
print("B basis_gates:", cfgB.basis_gates)
cmapA = A.coupling_map
cmapB = B.coupling_map


plot_coupling_map(A.num_qubits, None, cmapA.get_edges())
plot_coupling_map(B.num_qubits, None, cmapB.get_edges())