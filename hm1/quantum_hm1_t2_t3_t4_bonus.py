import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime, qiskit_aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator


def per_qubit_p1(counts, n):
    shots = sum(counts.values())
    p = np.zeros(n, dtype=float)
    for s, c in counts.items():                 # s like '0101' (qubit 0 is rightmost)
        for j, ch in enumerate(reversed(s)):    # map column 0 -> qubit 0
            if ch == '1':
                p[j] += c
    return p / max(shots, 1)

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
print("A basis gates:", A.configuration().basis_gates)
print("B basis gates:", B.configuration().basis_gates)

# quantum random number generator
k = 5
qc = qrng(k)
fig1 = qc.draw(output="mpl")
fig1.savefig("qrng_circuit.png", dpi=400, bbox_inches="tight")


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


# Draw the transpiled circuit
fig2 = isaA.draw(output="mpl")
fig2.savefig("isaA.png", dpi=400, bbox_inches="tight")
fig3 = isaB.draw(output="mpl")
fig3.savefig("isaB.png", dpi=400, bbox_inches="tight")


## Task 4

# Simulator
sim_sampler = BackendSamplerV2(backend=AerSimulator())
sim_counts = sim_sampler.run([isaA], shots=2000).result()[0].data.meas.get_counts()
print("Counts:", sim_counts)

# Hardware: target a specific backend
samplerA = SamplerV2(mode=A)
resultA = samplerA.run([isaA], shots=2000).result()
countsA = resultA[0].data.meas.get_counts() # {'010011': n, ...}
totalA = sum(countsA.values())
probsA = {bitstr: count / totalA for bitstr, count in countsA.items()}

samplerB = SamplerV2(mode=B)
resultB = samplerB.run([isaB], shots=2000).result()
countsB = resultB[0].data.meas.get_counts() # {'010011': n, ...}
totalB = sum(countsB.values())
probsB = {bitstr: count / totalB for bitstr, count in countsB.items()}


k = qc.num_qubits  # or isaA.num_qubits
p_sim = per_qubit_p1(sim_counts, k)
p_A = per_qubit_p1(countsA, k)
p_B = per_qubit_p1(countsB, k)

print("p_sim: ", p_sim)
print("p_A", p_A)
print("p_B", p_B)

x = np.arange(k); w = 0.33
fig4 = plt.figure()
plt.bar(x - w/3, p_sim, width=w/3, label="Aer (sampled)")
plt.bar(x, p_A,   width=w/3, label=A.name)
plt.bar(x + w/3, p_B,   width=w/3, label=B.name)
plt.xlabel("Qubit index"); plt.ylabel("Fraction of 1s (P(1))"); plt.title("Per-qubit bias")
plt.xticks(x, [f"q{j}" for j in range(k)]); plt.ylim(0, 1); plt.legend(); plt.tight_layout()

fig4.savefig("per_qubit_bias.png", dpi=300)




# bonus 1
print("--- Bonus 1 ---")

import numpy as np

def monobit_summary(counts, result,  k):
    shots = sum(counts.values())
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    p = M.mean(axis=0)                 # per-qubit fraction of 1s
    overall = float(p.mean())
    se = np.sqrt(0.25/shots)           # rough expected fluctuation for a fair coin
    suspect = np.abs(p - 0.5) > 3*se   # rule-of-thumb: outside ±3·SE
    return p, overall, se, suspect

pA, overallA, seA, flagA = monobit_summary(countsA, resultA, k)
print("A per-qubit P(1):", np.round(pA, 3), "overall:", round(overallA, 3), "SE~", round(seA, 4))
print("A suspect qubits:", np.where(flagA)[0].tolist())

pB, overallB, seB, flagB = monobit_summary(countsB, resultB, k)
print("B per-qubit P(1):", np.round(pB, 3), "overall:", round(overallB, 3), "SE~", round(seB, 4))
print("B suspect qubits:", np.where(flagB)[0].tolist())

# bonus 2
print("--- Bonus 2 ---")

def runs_fraction_per_qubit(counts, result, k):
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    flips = (M[1:] != M[:-1]).mean(axis=0)   # fraction of shot-to-shot flips per qubit
    return flips

flipsA = runs_fraction_per_qubit(countsA, resultA, k)
print("A runs (flip fraction) per qubit:", np.round(flipsA, 3))

flipsB = runs_fraction_per_qubit(countsB, resultB, k)
print("B runs (flip fraction) per qubit:", np.round(flipsB, 3))


# bonus 3

print("--- Bonus 3 ---")


def autocorr_lag1(counts, result, k):
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    X = M - M.mean(axis=0, keepdims=True)
    num = (X[1:]*X[:-1]).sum(axis=0)
    den = (X[:-1]**2).sum(axis=0)
    ac1 = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den>0)
    return ac1

ac1A = autocorr_lag1(countsA, resultA, k)
print("A lag-1 autocorr per qubit:", np.round(ac1A, 3))

ac1B = autocorr_lag1(countsB, resultB, k)
print("B lag-1 autocorr per qubit:", np.round(ac1B, 3))


# bonus 4

print("--- Bonus 4 ---")

import itertools

def interqubit_corr(counts, result, k):
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    X = M - M.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (len(M)-1)
    std = X.std(axis=0, ddof=1)
    R = cov / (std[:,None]*std[None,:])
    np.fill_diagonal(R, 1.0)
    return R

R_A = interqubit_corr(countsA, resultA, k)
flags = [(i,j,float(R_A[i,j])) for i,j in itertools.combinations(range(k),2) if abs(R_A[i,j])>0.1]
print("A suspicious pairs:", flags[:10])

R_B = interqubit_corr(countsB, resultB, k)
flags = [(i,j,float(R_B[i,j])) for i,j in itertools.combinations(range(k),2) if abs(R_B[i,j])>0.1]
print("B suspicious pairs:", flags[:10])

print("--- Bonus END ---")