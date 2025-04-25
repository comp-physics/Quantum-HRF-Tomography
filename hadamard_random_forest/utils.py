"""
Helper functions for estimating quantum state properties such as entanglement, magic, and state overlap.
"""

from __future__ import annotations
from typing import List, Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, HGate, CSwapGate
from qiskit.quantum_info import PauliList, Statevector

__all__ = [
    "partial_transpose",
    "trace_norm",
    "logarithmic_negativity",
    "generate_integer_list",
    "integer_to_pauli",
    "stabilizer_entropy",
    "swap_test",
]

### ============================== Quantum entanglement helper functions ==============================

def partial_transpose(density_op: np.ndarray, num_sysA: int, num_qubits: int) -> np.ndarray:
    r"""
    Calculate the partial transpose :math:`\rho^{T_A}` of a bipartite quantum state.

    Args:
        density_op: Density matrix of shape (2**num_qubits, 2**num_qubits).
        num_sysA: Number of qubits in subsystem A (first subsystem).
        num_qubits: Total number of qubits in the system.

    Returns:
        Partially transposed density matrix of shape (2**num_qubits, 2**num_qubits).
    """
    # Dimensions for subsystem A and B.
    dim_A = 2 ** num_sysA
    dim_B = 2 ** (num_qubits - num_sysA)
    
    # Reshape the density matrix to a tensor of shape (dim_A, dim_B, dim_A, dim_B)
    tensor = np.reshape(density_op, (dim_A, dim_B, dim_A, dim_B))
    # Perform the partial transpose on subsystem A by swapping the first and third indices.
    tensor_pt = np.transpose(tensor, (2, 1, 0, 3))
    # Reshape back to a square matrix.
    return np.reshape(tensor_pt, (2 ** num_qubits, 2 ** num_qubits))


def trace_norm(
    matrix: np.ndarray
) -> float:
    """
    Compute the trace (nuclear) norm of a matrix.

    Args:
        matrix: A square or rectangular numpy array.

    Returns:
        The sum of singular values (nuclear norm).
    """
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    return float(np.sum(singular_values))


def logarithmic_negativity(
    density_op: np.ndarray,
    num_sysA: int,
    num_qubits: int
) -> float:
    """
    Compute the logarithmic negativity of a bipartite quantum state.

    Uses the trace norm of the partial transpose:
        LN = log2(||rho^{T_A}||_1)

    Args:
        density_op: Density matrix of shape (2**num_qubits, 2**num_qubits).
        num_sysA: Number of qubits in subsystem A.
        num_qubits: Total number of qubits.

    Returns:
        The logarithmic negativity as a float.
    """
    rho_pt = partial_transpose(density_op, num_sysA, num_qubits)
    tn = trace_norm(rho_pt)
    return np.log2(tn)


### ============================== Quantum magic helper functions ==============================

def generate_integer_list(
    index: int,
    num_qubits: int
) -> np.ndarray:
    """
    Convert an integer into its base-4 representation of length num_qubits.

    Args:
        index: Integer in [0, 4**num_qubits) representing Pauli operator.
        num_qubits: Number of qubits (digits in base-4 representation).

    Returns:
        A numpy array of shape (num_qubits,) with entries 0->I, 1->X, 2->Z, 3->Y. 
        integer list [[0,0,...,0], [0,0,...,1], ..., [3,3,...,3]]
    """
    if index < 0 or index >= 4**num_qubits:
        raise ValueError(f"Index must be in [0, {4**num_qubits}), got {index}.")
    digits = np.zeros(num_qubits, dtype=int)
    for pos in range(num_qubits - 1, -1, -1):
        digits[pos] = index % 4
        index //= 4
    return digits


def integer_to_pauli(
    pauli_indices: Sequence[int]
) -> str:
    """
    Convert a sequence of base-4 Pauli indices into a Pauli string.

    Args:
        pauli_indices: Sequence of ints (0,1,2,3) of length num_qubits.

    Returns:
        A string of length num_qubits over the alphabet {I,X,Z,Y}, Pauli string list ['II...I', 'II...X', ..., 'YY...Y'].
    """
    mapping = {0: "I", 1: "X", 2: "Z", 3: "Y"}
    try:
        return ''.join(mapping[i] for i in pauli_indices)
    except KeyError as e:
        raise ValueError(f"Invalid Pauli index {e.args[0]}; must be 0-3.")

def stabilizer_entropy(
    alpha: float,
    psi: np.ndarray
) -> float:
    """
    Calculate the exact alpha-Rényi Stabilizer Entropy for a pure state.

    Args:
        alpha: entropic index.
        psi: Statevector as a 1D numpy array of length 2**num_qubits.

    Returns:
        The stabilizer entropy S_alpha(psi).
    """
    num_qubits = int(np.log2(len(psi)))

    # Generate the n-qubit Pauli group including 4^n combinations of Pauli strings
    integer_pauli = [generate_integer_list(k, num_qubits) for k in range(4**num_qubits)]
    string_pauli = [integer_to_pauli(integer_pauli[k]) for k in range(4**num_qubits)]
    expval_list = []
    
    # Calculate the expectation value for each Pauli string
    for op in PauliList(string_pauli):
        expval_list.append(psi.expectation_value(op))
    
    # Calculate the moment for stabilizer entropy (SE)
    expval_list = np.array(expval_list)
    moment = np.sum(expval_list**(2*alpha)/(2**num_qubits))
    
    # Calculate final stabilizer entropy
    entropy = np.log2(moment)/(1-alpha) 

    return entropy


### ============================== Quantum state overlap helper functions ==============================

def swap_test(
    psi1: np.ndarray,
    psi2: np.ndarray
) -> qiskit.QuantumCircuit:
    """
    Build a SWAP test circuit to compare two statevectors.

    Args:
        psi1: Statevector of first state as a 1D numpy array of length 2**num_qubits.
        psi2: Statevector of second state (same length as psi1).

    Returns:
        A QuantumCircuit implementing the SWAP test, with:
          - Qubit 0 as ancilla.
          - Qubits 1..num_qubits for psi1 preparation.
          - Qubits num_qubits+1..2*num_qubits for psi2 preparation.
          - Final Hadamard on ancilla. 
    """
    # Determine number of qubits in each state register
    num_qubits = int(np.log2(psi1.size))
    if psi1.shape != psi2.shape:
        raise ValueError("Statevectors must have the same shape.")

    # Create circuit: 1 ancilla + two state registers
    qc = QuantumCircuit(2 * num_qubits + 1)

    # Prepare ancilla and apply first Hadamard
    qc.append(HGate(), [0])

    # Prepare state registers
    prep1 = StatePreparation(psi1)
    prep2 = StatePreparation(psi2)
    qc.compose(prep1, list(range(1, 1 + num_qubits)), inplace=True)
    qc.compose(prep2, list(range(1 + num_qubits, 1 + 2 * num_qubits)), inplace=True)

    # Controlled SWAPs between corresponding qubits
    for i in range(num_qubits):
        qc.append(CSwapGate(), [0, 1 + i, 1 + num_qubits + i])

    # Final Hadamard on ancilla
    qc.append(HGate(), [0])

    return qc



