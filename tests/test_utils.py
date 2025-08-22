"""
Test suite for hadamard_random_forest.utils module.

Tests are based on examples from tutorial notebooks:
- 03a_estimate_entanglement.ipynb for entanglement calculations
- 03b_estimate_magic.ipynb for magic calculations  
- 03c_estimate_state_overlap.ipynb for state overlap and swap test
"""

import unittest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Import the functions to test
import hadamard_random_forest as hrf
from hadamard_random_forest.utils import (
    random_statevector,
    partial_transpose,
    trace_norm,
    logarithmic_negativity,
    generate_integer_list,
    integer_to_pauli,
    stabilizer_entropy,
    swap_test
)

# Import Qiskit components
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, PauliList
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator


class TestQuantumEntanglement(unittest.TestCase):
    """Test quantum entanglement helper functions."""
    
    def setUp(self):
        """Set up test cases with known quantum states."""
        # Fix random seed for reproducibility
        np.random.seed(42)
        
    def test_random_statevector_shape_and_normalization(self):
        """Test random statevector generation."""
        for num_qubits in [1, 2, 3, 4]:
            sv = random_statevector(num_qubits)
            # Check shape
            self.assertEqual(sv.shape, (2**num_qubits,))
            # Check normalization
            self.assertAlmostEqual(np.linalg.norm(sv), 1.0, places=10)
            # Check complex values
            self.assertTrue(np.iscomplexobj(sv))
    
    def test_partial_transpose_bell_state(self):
        """Test partial transpose on Bell state from tutorial 03a."""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell_sv = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(bell_sv, bell_sv.conj())
        
        # Partial transpose with respect to first qubit
        rho_pt = partial_transpose(rho, num_sysA=1, num_qubits=2)
        
        # For Bell state, partial transpose should have negative eigenvalue
        eigenvals = np.linalg.eigvals(rho_pt)
        self.assertTrue(np.any(eigenvals < -1e-10))  # Some eigenvalue should be negative
        
    def test_partial_transpose_separable_state(self):
        """Test partial transpose on separable state from tutorial 03a."""
        # Separable state |ψ⟩ = |ψ1⟩ ⊗ |ψ2⟩
        psi1 = random_statevector(2)  # 2-qubit state for subsystem A
        psi2 = random_statevector(3)  # 3-qubit state for subsystem B
        sv = np.kron(psi1, psi2)
        rho = np.outer(sv, sv.conj())
        
        # Partial transpose should preserve positive semi-definiteness for separable states
        rho_pt = partial_transpose(rho, num_sysA=2, num_qubits=5)
        eigenvals = np.linalg.eigvals(rho_pt)
        self.assertTrue(np.all(eigenvals >= -1e-10))  # All eigenvalues should be non-negative
        
    def test_trace_norm_properties(self):
        """Test trace norm computation."""
        # Test with random matrices
        for size in [2, 4, 8]:
            matrix = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            tn = trace_norm(matrix)
            
            # Trace norm should be positive
            self.assertGreaterEqual(tn, 0)
            # Trace norm should equal sum of singular values
            _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
            expected_tn = np.sum(singular_values)
            self.assertAlmostEqual(tn, expected_tn, places=10)
    
    def test_logarithmic_negativity_bell_state(self):
        """Test logarithmic negativity on Bell state (tutorial 03a example)."""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell_sv = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(bell_sv, bell_sv.conj())
        
        # From tutorial: LN should be approximately 1 for Bell state
        LN = logarithmic_negativity(rho, num_sysA=1, num_qubits=2)
        self.assertAlmostEqual(LN, 1.0, places=10)
        
    def test_logarithmic_negativity_separable_state(self):
        """Test logarithmic negativity on separable state (tutorial 03a example)."""
        # Create separable state as in tutorial
        num_sysA = 2
        num_sysB = 3
        num_qubits = num_sysA + num_sysB
        psi1 = random_statevector(num_sysA)
        psi2 = random_statevector(num_sysB)
        sv = np.kron(psi1, psi2)
        rho = np.outer(sv, sv.conj())
        
        # From tutorial: LN should be approximately 0 for separable states
        LN = logarithmic_negativity(rho, num_sysA, num_qubits)
        self.assertAlmostEqual(LN, 0.0, places=10)
    
    def test_partial_transpose_edge_cases(self):
        """Test partial transpose edge cases."""
        # Single qubit case
        sv = random_statevector(1)
        rho = np.outer(sv, sv.conj())
        rho_pt = partial_transpose(rho, num_sysA=1, num_qubits=1)
        # For single qubit, partial transpose is just transpose
        np.testing.assert_array_almost_equal(rho_pt, rho.T)
        
    def test_logarithmic_negativity_input_validation(self):
        """Test input validation for logarithmic negativity."""
        # Create valid 2-qubit density matrix
        sv = random_statevector(2)
        rho = np.outer(sv, sv.conj())
        
        # Valid input should work
        LN = logarithmic_negativity(rho, num_sysA=1, num_qubits=2)
        self.assertIsInstance(LN, float)


class TestQuantumMagic(unittest.TestCase):
    """Test quantum magic helper functions."""
    
    def setUp(self):
        """Set up test cases."""
        np.random.seed(42)
        
    def test_generate_integer_list_conversion(self):
        """Test integer to base-4 conversion."""
        # Test cases from tutorial understanding
        result = generate_integer_list(0, 2)
        np.testing.assert_array_equal(result, [0, 0])  # I,I
        
        result = generate_integer_list(1, 2)
        np.testing.assert_array_equal(result, [0, 1])  # I,X
        
        result = generate_integer_list(15, 2)  # 15 = 3*4 + 3
        np.testing.assert_array_equal(result, [3, 3])  # Y,Y
        
    def test_generate_integer_list_edge_cases(self):
        """Test edge cases for generate_integer_list."""
        # Test boundary values
        result = generate_integer_list(0, 1)
        np.testing.assert_array_equal(result, [0])
        
        result = generate_integer_list(3, 1)
        np.testing.assert_array_equal(result, [3])
        
        # Test invalid inputs
        with self.assertRaises(ValueError):
            generate_integer_list(-1, 2)
        with self.assertRaises(ValueError):
            generate_integer_list(16, 2)  # 4^2 = 16, so max index is 15
            
    def test_integer_to_pauli_conversion(self):
        """Test Pauli index to string conversion."""
        # Test all single Pauli operators
        self.assertEqual(integer_to_pauli([0]), "I")
        self.assertEqual(integer_to_pauli([1]), "X")
        self.assertEqual(integer_to_pauli([2]), "Z")
        self.assertEqual(integer_to_pauli([3]), "Y")
        
        # Test multi-qubit Pauli strings
        self.assertEqual(integer_to_pauli([0, 0]), "II")
        self.assertEqual(integer_to_pauli([0, 1]), "IX")
        self.assertEqual(integer_to_pauli([3, 3]), "YY")
        
    def test_integer_to_pauli_invalid_input(self):
        """Test invalid input for integer_to_pauli."""
        with self.assertRaises(ValueError):
            integer_to_pauli([4])  # Invalid Pauli index
        with self.assertRaises(ValueError):
            integer_to_pauli([-1])  # Invalid Pauli index
            
    def test_stabilizer_entropy_clifford_circuit(self):
        """Test stabilizer entropy on Clifford circuit (tutorial 03b example)."""
        # Create Clifford circuit from tutorial 03b
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.s(2)
        qc.cx(0, 1)
        qc.cx(1, 3)
        qc.s(0)
        qc.h(3)
        qc.cx(0, 2)
        qc.cx(3, 2)
        qc.cx(1, 2)
        
        psi = Statevector(qc)
        alpha = 2
        SE = stabilizer_entropy(alpha, psi)
        
        # From tutorial: stabilizer states should have SE ≈ 0
        self.assertAlmostEqual(SE, 0.0, places=10)
        
    def test_stabilizer_entropy_t_gate_circuit(self):
        """Test stabilizer entropy with T-gate (tutorial 03b example)."""
        # Create circuit with T-gate from tutorial 03b
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.s(2)
        qc.cx(0, 1)
        qc.cx(1, 3)
        qc.s(0)
        qc.h(3)
        qc.cx(0, 2)
        qc.cx(3, 2)
        qc.cx(1, 2)
        qc.barrier()
        qc.t(3)  # Add T-gate
        
        psi = Statevector(qc)
        alpha = 2
        SE = stabilizer_entropy(alpha, psi)
        
        # From tutorial: T-gate should make SE > 0
        self.assertGreater(SE, 0.1)  # Should be around 0.415 from tutorial
        
    def test_stabilizer_entropy_magic_state(self):
        """Test stabilizer entropy on |+⟩ + T gate (tutorial 03b example)."""
        # Create √T|+⟩ state from tutorial 03b
        qc = QuantumCircuit(1)
        qc.h(0)  # |+⟩ state
        qc.rz(np.pi/8, 0)  # √T gate (π/8 rotation)
        
        psi = Statevector(qc)
        alpha = 2
        SE = stabilizer_entropy(alpha, psi)
        
        # From tutorial: should be 3 - log2(7) ≈ 0.1926
        expected = 3 - np.log2(7)
        self.assertAlmostEqual(SE, expected, places=5)
        
    def test_stabilizer_entropy_single_qubit_cases(self):
        """Test stabilizer entropy on single qubit states."""
        # |0⟩ state (stabilizer)
        qc = QuantumCircuit(1)
        psi = Statevector(qc)
        alpha = 2
        SE = stabilizer_entropy(alpha, psi)
        self.assertAlmostEqual(SE, 0.0, places=10)
        
        # |+⟩ state (stabilizer)
        qc = QuantumCircuit(1)
        qc.h(0)
        psi = Statevector(qc)
        SE = stabilizer_entropy(alpha, psi)
        self.assertAlmostEqual(SE, 0.0, places=10)
        
    def test_stabilizer_entropy_numpy_array_input(self):
        """Test stabilizer entropy with numpy array input."""
        alpha = 2
        
        # Test |0⟩ state as numpy array
        psi_array = np.array([1.0, 0.0], dtype=complex)
        SE_array = stabilizer_entropy(alpha, psi_array)
        
        # Test same state as Statevector
        qc = QuantumCircuit(1)
        psi_statevector = Statevector(qc)
        SE_statevector = stabilizer_entropy(alpha, psi_statevector)
        
        # Results should be identical
        self.assertAlmostEqual(SE_array, SE_statevector, places=10)
        self.assertAlmostEqual(SE_array, 0.0, places=10)
        
        # Test Bell state as numpy array
        bell_array = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        SE_bell = stabilizer_entropy(alpha, bell_array)
        self.assertAlmostEqual(SE_bell, 0.0, places=10)  # Bell state is stabilizer


class TestSwapTest(unittest.TestCase):
    """Test SWAP test implementation for state overlap."""
    
    def setUp(self):
        """Set up test backend."""
        self.backend = AerSimulator()
        np.random.seed(42)
        
    def test_swap_test_identical_states(self):
        """Test SWAP test with identical states."""
        # Create identical random states
        state = random_statevector(2)
        
        # Mock the backend run to avoid actual execution
        with patch.object(self.backend, 'run') as mock_run:
            # Mock result for identical states (should get mostly '0' outcomes)
            mock_result = MagicMock()
            mock_result.get_counts.return_value = {'0': 950, '1': 50}  # Mostly |0⟩
            mock_run.return_value.result.return_value = mock_result
            
            qc, overlap_est, overlap_exact = swap_test(state, state, self.backend, shots=1000)
            
            # For identical states, overlap should be 1
            self.assertAlmostEqual(overlap_exact, 1.0, places=10)
            # Estimated overlap should be close to 1 (1 - 2*0.05 = 0.9)
            self.assertAlmostEqual(overlap_est, 0.9, places=1)
            
    def test_swap_test_orthogonal_states(self):
        """Test SWAP test with orthogonal states."""
        # Create orthogonal states
        state1 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        state2 = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        
        with patch.object(self.backend, 'run') as mock_run:
            # Mock result for orthogonal states (should get 50-50 outcomes)
            mock_result = MagicMock()
            mock_result.get_counts.return_value = {'0': 500, '1': 500}  # Equal outcomes
            mock_run.return_value.result.return_value = mock_result
            
            qc, overlap_est, overlap_exact = swap_test(state1, state2, self.backend, shots=1000)
            
            # For orthogonal states, overlap should be 0
            self.assertAlmostEqual(overlap_exact, 0.0, places=10)
            # Estimated overlap should be 0 (1 - 2*0.5 = 0)
            self.assertAlmostEqual(overlap_est, 0.0, places=1)
            
    def test_swap_test_circuit_inputs(self):
        """Test SWAP test with QuantumCircuit inputs."""
        # Create circuits preparing states
        qc1 = QuantumCircuit(2)
        qc1.h(0)  # |+0⟩ state
        
        qc2 = QuantumCircuit(2)
        qc2.x(1)  # |01⟩ state
        
        with patch.object(self.backend, 'run') as mock_run:
            mock_result = MagicMock()
            mock_result.get_counts.return_value = {'0': 600, '1': 400}
            mock_run.return_value.result.return_value = mock_result
            
            qc, overlap_est, overlap_exact = swap_test(qc1, qc2, self.backend, shots=1000)
            
            # Should execute without error
            self.assertIsInstance(qc, QuantumCircuit)
            self.assertIsInstance(overlap_est, (int, float))
            self.assertIsInstance(overlap_exact, (int, float))
            
    def test_swap_test_input_validation(self):
        """Test input validation for swap_test."""
        state = random_statevector(2)
        
        # Test mismatched state sizes
        state_wrong_size = random_statevector(1)
        with self.assertRaises(ValueError):
            swap_test(state, state_wrong_size, self.backend, shots=1000)
            
        # Test invalid state type
        with self.assertRaises(TypeError):
            swap_test("invalid", state, self.backend, shots=1000)
            
    def test_swap_test_circuit_structure(self):
        """Test SWAP test circuit structure."""
        state1 = random_statevector(2)
        state2 = random_statevector(2)
        
        with patch.object(self.backend, 'run') as mock_run:
            mock_result = MagicMock()
            mock_result.get_counts.return_value = {'0': 500, '1': 500}
            mock_run.return_value.result.return_value = mock_result
            
            qc, _, _ = swap_test(state1, state2, self.backend, shots=1000)
            
            # Check circuit structure
            self.assertEqual(qc.num_qubits, 5)  # 2*2 + 1 ancilla
            self.assertEqual(qc.num_clbits, 1)  # 1 measurement
            
            # Check that circuit contains H and CSWAP gates
            gate_names = [instr.operation.name for instr in qc.data]
            self.assertIn('h', gate_names)
            self.assertIn('cswap', gate_names)
            self.assertIn('measure', gate_names)


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests combining multiple utils functions."""
    
    def setUp(self):
        """Set up for integration tests."""
        np.random.seed(42)
        
    def test_entanglement_and_magic_correlation(self):
        """Test correlation between entanglement and magic measures."""
        # Create a series of states with varying entanglement/magic
        states = []
        
        # Product state (separable, stabilizer)
        qc1 = QuantumCircuit(2)
        states.append(Statevector(qc1))
        
        # Bell state (entangled, stabilizer)
        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)
        states.append(Statevector(qc2))
        
        # Bell + T gate (entangled, non-stabilizer)
        qc3 = QuantumCircuit(2)
        qc3.h(0)
        qc3.cx(0, 1)
        qc3.t(0)
        states.append(Statevector(qc3))
        
        entanglements = []
        magics = []
        
        for state in states:
            rho = DensityMatrix(state)
            LN = logarithmic_negativity(rho.data, num_sysA=1, num_qubits=2)
            SE = stabilizer_entropy(2, state)
            entanglements.append(LN)
            magics.append(SE)
            
        # Verify expected properties
        # Product state: LN ≈ 0, SE ≈ 0
        self.assertAlmostEqual(entanglements[0], 0.0, places=10)
        self.assertAlmostEqual(magics[0], 0.0, places=10)
        
        # Bell state: LN = 1, SE ≈ 0
        self.assertAlmostEqual(entanglements[1], 1.0, places=10)
        self.assertAlmostEqual(magics[1], 0.0, places=10)
        
        # Bell + T: LN = 1, SE > 0
        self.assertAlmostEqual(entanglements[2], 1.0, places=5)
        self.assertGreater(magics[2], 0.1)
        
    def test_pauli_generation_completeness(self):
        """Test that Pauli generation covers all operators."""
        num_qubits = 2
        total_paulis = 4**num_qubits  # 16 for 2 qubits
        
        pauli_strings = []
        for k in range(total_paulis):
            integer_list = generate_integer_list(k, num_qubits)
            pauli_string = integer_to_pauli(integer_list)
            pauli_strings.append(pauli_string)
            
        # Check we have all unique Pauli strings
        unique_paulis = set(pauli_strings)
        self.assertEqual(len(unique_paulis), total_paulis)
        
        # Check specific strings exist
        self.assertIn("II", pauli_strings)
        self.assertIn("IX", pauli_strings)
        self.assertIn("YY", pauli_strings)
        
    def test_utils_numerical_stability(self):
        """Test numerical stability of utils functions."""
        # Test with very small numbers
        small_sv = np.array([1e-10, 1e-10, 1e-10, np.sqrt(1 - 3e-20)], dtype=complex)
        small_sv = small_sv / np.linalg.norm(small_sv)
        
        # Should not raise numerical errors
        rho = np.outer(small_sv, small_sv.conj())
        LN = logarithmic_negativity(rho, num_sysA=1, num_qubits=2)
        self.assertIsInstance(LN, float)
        self.assertFalse(np.isnan(LN))
        
        SE = stabilizer_entropy(2, Statevector(small_sv))
        self.assertIsInstance(SE, float)
        self.assertFalse(np.isnan(SE))


if __name__ == '__main__':
    # Suppress warnings during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    unittest.main(verbosity=2)