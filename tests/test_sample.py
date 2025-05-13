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
        num_qubits = 3
        base_circuit = random_circuit(num_qubits,3, measure=False)
        circuits = get_circuits(num_qubits, base_circuit)
        self.assertEqual(len(circuits), 4)
        self.assertEqual(circuits[0].num_qubits, 3)
        self.assertEqual(circuits[1].num_qubits, 3)

    def test_get_samples(self):
        num_qubits = 3
        sampler = Aer_Sampler()
        base_circuit = real_amplitudes(num_qubits)
        parameters = np.random.rand(12)
        circuits = get_circuits(num_qubits, base_circuit)
        _ = get_samples(num_qubits, sampler, circuits, parameters)  

    def test_get_statevector(self):
        """
        Test that get_statevector returns a valid statevector.
        """
        num_qubits = 3
        num_trees = 5
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits+1)]
        statevector = get_statevector(num_qubits, num_trees, samples, save_tree=False)
        self.assertIsInstance(statevector, np.ndarray)
        self.assertEqual(statevector.shape, (2**num_qubits,))

if __name__ == '__main__':
    unittest.main()