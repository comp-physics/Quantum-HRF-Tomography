import unittest
import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit_aer.primitives import Sampler as Aer_Sampler
from qiskit.circuit.library import real_amplitudes
from hadamard_random_forest.sample import (
    get_statevector,
    get_circuits,
    get_samples
)

class TestSample(unittest.TestCase):

    def test_get_circuits(self):
        """Test the get_circuits function."""
        num_qubits = 3
        base_circuit = real_amplitudes(num_qubits)
        circuits = get_circuits(num_qubits, base_circuit)
        
        # Original tests
        self.assertIsInstance(circuits, list)
        self.assertEqual(len(circuits), 4)
        
        # Enhanced tests
        for circuit in circuits:
            self.assertEqual(circuit.num_qubits, num_qubits)
            self.assertIsNotNone(circuit)

    def test_get_samples(self):
        """Test the get_samples function."""
        num_qubits = 3
        sampler = Aer_Sampler()
        base_circuit = real_amplitudes(num_qubits)
        circuits = get_circuits(num_qubits, base_circuit)
        parameters = np.random.rand(base_circuit.num_parameters)
        samples = get_samples(num_qubits, sampler, circuits, parameters)
        
        # Original tests
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 4)
        
        # Enhanced tests
        for sample in samples:
            self.assertIsInstance(sample, np.ndarray)
            self.assertEqual(sample.shape, (8,))  # 2^3 = 8
            self.assertTrue(np.all(sample >= 0))  # Non-negative probabilities
            self.assertAlmostEqual(np.sum(sample), 1.0, places=10)  # Normalized

    def test_get_statevector(self):
        """Test the get_statevector function."""
        num_qubits = 3
        num_trees = 5
        samples = [np.random.rand(8) for _ in range(4)]
        
        # Normalize samples to represent valid probability distributions
        samples = [s / np.sum(s) for s in samples]
        
        statevector = get_statevector(num_qubits, num_trees, samples, save_tree=False)
        
        # Original tests
        self.assertIsInstance(statevector, np.ndarray)
        self.assertEqual(statevector.shape, (8,))
        
        # Enhanced tests
        self.assertTrue(np.all(np.isfinite(statevector)))  # All elements finite
        self.assertGreater(np.linalg.norm(statevector), 0)  # Non-zero result

    def test_integration(self):
        """Test the complete workflow integration."""
        num_qubits = 2  # Smaller for faster testing
        num_trees = 3
        sampler = Aer_Sampler()
        base_circuit = real_amplitudes(num_qubits)
        parameters = np.random.rand(base_circuit.num_parameters)
        
        # Complete workflow
        circuits = get_circuits(num_qubits, base_circuit)
        samples = get_samples(num_qubits, sampler, circuits, parameters)
        statevector = get_statevector(num_qubits, num_trees, samples, save_tree=False)
        
        # Validate end-to-end
        self.assertEqual(statevector.shape, (2**num_qubits,))
        self.assertTrue(np.all(np.isfinite(statevector)))

if __name__ == '__main__':
    unittest.main()