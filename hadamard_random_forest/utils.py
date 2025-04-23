r"""
Helper functions for estimating quantum state properties such as entanglement, magic, and state overlap.
"""

import numpy as np
from qiskit.quantum_info import PauliList

__all__ = [
    "generate_integer_list",
    "integer_to_pauli",
    "stabilizer_entropy"
]


### ============================== Quantum entanglement helper functions ==============================

def partial_transpose(density_op: np.ndarray, n: int, n_qubits: int) -> np.ndarray:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of a bipartite quantum state.
    
    Args:
        density_op: Density matrix as a numpy array with shape (2^n_qubits, 2^n_qubits).
        n: Number of qubits in subsystem A.
        n_qubits: Total number of qubits.
        
    Returns:
        The partially transposed density matrix as a numpy array.
    """
    # Dimensions for subsystem A and B.
    dim_A = 2 ** n
    dim_B = 2 ** (n_qubits - n)
    
    # Reshape the density matrix to a tensor of shape (dim_A, dim_B, dim_A, dim_B)
    tensor = np.reshape(density_op, (dim_A, dim_B, dim_A, dim_B))
    # Perform the partial transpose on subsystem A by swapping the first and third indices.
    tensor_pt = np.transpose(tensor, (2, 1, 0, 3))
    # Reshape back to a square matrix.
    return np.reshape(tensor_pt, (2 ** n_qubits, 2 ** n_qubits))


### ============================== Quantum magic helper functions ==============================

def generate_integer_list(k, num_qubits):
    """
    Generate a n-qubit Pauli list in integer format with 4^n elements.
    k: Pauli type. 0->'I', 1->'X', 2->'Z', 3->'Y'
    num_qubits: number of qubits

    returns an integer list [[0,0,...,0], [0,0,...,1], ..., [3,3,...,3]]
    """
    if k == 0:
        return np.zeros(num_qubits, dtype=int)
    digits = np.zeros(num_qubits, dtype=int)
    counter = 0
    while k:
        digits[counter] = int(k % 4)
        k //= 4
        counter += 1
    return digits[::-1]

def integer_to_pauli(pauli):
    """
    Convert the integer Pauli list to string format.
    pauli: 4^n n-qubit Pauli strings

    returns the Pauli string list ['II...I', 'II...X', ..., 'YY...Y']
    """
    string = []
    for i in (range(len(pauli))):
        if(pauli[i] == 0):
            string_type = "I"
        elif(pauli[i] == 1):
            string_type = "X"
        elif(pauli[i] == 2):
            string_type = "Z" 
        elif(pauli[i] == 3):
            string_type = "Y"
        string.append(string_type)
    return "".join(string)

def stabilizer_entropy(alpha, psi):
    """
    Calculate the exact alpha-Rényi Stabilizer Entropy for infinite number of samples. 
    alpha: entropic index
    psi: quantum state in statevector format
    
    returns the stabilizer entropy
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

### ============================== Visualization functions for tree strcuture ==============================