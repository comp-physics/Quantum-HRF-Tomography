import unittest
import numpy as np
from hadamard_random_forest.random_forest import (generate_random_forest)


class TestRandomForest(unittest.TestCase):

    def test_generate_random_forest(self):
        """
        Test that generate_random_forest returns a valid result.
        """
        num_qubits = 3
        num_trees = 5
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits+1)]
        result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2**num_qubits,))


if __name__ == '__main__':
    unittest.main()

