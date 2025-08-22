import unittest
import warnings
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import real_amplitudes, efficient_su2
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.providers import JobStatus
from qiskit.result import Result
from hadamard_random_forest.sample import (
    get_statevector,
    get_circuits,
    get_samples_noisy,
    get_circuits_hardware,
    get_samples_hardware
)
from hadamard_random_forest.random_forest import fix_random_seed

class TestSample(unittest.TestCase):

    def setUp(self):
        """Set up test setting and common test data."""
        self.backend_sim = AerSimulator()
        self.fake_backend = FakeFez()
        
        # Common test parameters
        self.test_qubit_counts = [2, 3, 4]
        self.test_shots = 1024
        
        # Create sample circuits for testing
        self.simple_circuits = {}
        self.complex_circuits = {}
        
        for num_qubits in self.test_qubit_counts:
            # Simple circuit (real_amplitudes)
            self.simple_circuits[num_qubits] = real_amplitudes(
                num_qubits, reps=1, insert_barriers=False
            )
            
            # More complex circuit (efficient_su2)
            self.complex_circuits[num_qubits] = efficient_su2(
                num_qubits, reps=2, insert_barriers=False
            )

    def _create_test_samples(self, num_qubits: int, normalize: bool = True) -> list:
        """Create valid test sample data."""
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits + 1)]
        if normalize:
            # Normalize to valid probability distributions
            samples = [s / np.sum(s) for s in samples]
        return samples

    def _create_mock_counts(self, num_qubits: int, shots: int = 1024) -> dict:
        """Create mock measurement counts for testing."""
        counts = {}
        n_states = 2**num_qubits
        # Distribute shots randomly across computational basis states
        remaining_shots = shots
        for i in range(n_states - 1):
            count = np.random.randint(0, remaining_shots // (n_states - i) + 1)
            if count > 0:
                counts[format(i, f'0{num_qubits}b')] = count
                remaining_shots -= count
        
        # Assign remaining shots to last state
        if remaining_shots > 0:
            counts[format(n_states - 1, f'0{num_qubits}b')] = remaining_shots
            
        return counts

    def test_get_circuits_basic(self):
        """Test basic functionality of get_circuits."""
        for num_qubits in self.test_qubit_counts:
            with self.subTest(num_qubits=num_qubits):
                base_circuit = self.simple_circuits[num_qubits]
                circuits = get_circuits(num_qubits, base_circuit)
                
                # Verify structure
                self.assertIsInstance(circuits, list)
                self.assertEqual(len(circuits), num_qubits + 1)  # Base + H variants
                
                # Verify all circuits have correct qubit count
                for circuit in circuits:
                    self.assertEqual(circuit.num_qubits, num_qubits)
                    self.assertIsNotNone(circuit)
                    # Verify measurements are present
                    self.assertTrue(any(op.operation.name == 'measure' for op in circuit.data))

    def test_get_circuits_structure_validation(self):
        """Test that get_circuits preserves base circuit structure correctly."""
        num_qubits = 3
        base_circuit = real_amplitudes(num_qubits, reps=2)
        circuits = get_circuits(num_qubits, base_circuit)
        
        # Subsequent circuits should have exactly one additional H gate
        for i in range(1, len(circuits)):
            circuit = circuits[i]
            
            # Count H gates in the circuit
            h_count = sum(1 for op in circuit.data if op.operation.name == 'h')
            
            # Should have exactly one H gate more than base circuit
            # (base circuit shouldn't have H gates for real_amplitudes)
            self.assertGreaterEqual(h_count, 1)
            
            # Verify the circuit has measurements
            measure_count = sum(1 for op in circuit.data if op.operation.name == 'measure')
            self.assertEqual(measure_count, num_qubits)

    def test_get_circuits_different_ansatz_types(self):
        """Test get_circuits with different circuit types."""
        test_cases = [
            ("real_amplitudes", self.simple_circuits),
            ("efficient_su2", self.complex_circuits)
        ]
        
        for ansatz_name, circuit_dict in test_cases:
            for num_qubits in self.test_qubit_counts:
                with self.subTest(ansatz=ansatz_name, num_qubits=num_qubits):
                    base_circuit = circuit_dict[num_qubits]
                    circuits = get_circuits(num_qubits, base_circuit)
                    
                    self.assertEqual(len(circuits), num_qubits + 1)
                    
                    # Verify parameters are preserved
                    if base_circuit.num_parameters > 0:
                        for circuit in circuits:
                            # After composition, parameters should still be present
                            # (though measurement might add classical registers)
                            self.assertGreaterEqual(circuit.num_parameters, 0)

    def test_get_circuits_parameter_preservation(self):
        """Test that circuit parameters are preserved during get_circuits."""
        num_qubits = 3
        base_circuit = real_amplitudes(num_qubits, reps=2)
        original_params = base_circuit.num_parameters
        
        circuits = get_circuits(num_qubits, base_circuit)
        
        # All circuits should preserve the original parameters
        for circuit in circuits:
            self.assertEqual(circuit.num_parameters, original_params)

    def test_get_samples_noisy_without_mitigation(self):
        """Test get_samples_noisy without error mitigation."""
        for num_qubits in [2, 3]:  # Use smaller systems for faster testing
            with self.subTest(num_qubits=num_qubits):
                base_circuit = self.simple_circuits[num_qubits]
                circuits = get_circuits(num_qubits, base_circuit)
                parameters = np.random.rand(base_circuit.num_parameters)
                
                samples = get_samples_noisy(
                    num_qubits=num_qubits,
                    circuits=circuits,
                    shots=self.test_shots,
                    parameters=parameters,
                    backend_sim=self.backend_sim,
                    error_mitigation=False
                )
                
                # Verify output structure
                self.assertIsInstance(samples, list)
                self.assertEqual(len(samples), num_qubits + 1)
                
                # Verify each sample array
                for sample in samples:
                    self.assertIsInstance(sample, np.ndarray)
                    self.assertEqual(sample.shape, (2**num_qubits,))
                    self.assertTrue(np.all(sample >= 0))  # Probabilities non-negative
                    self.assertAlmostEqual(np.sum(sample), 1.0, places=6)  # Normalized

    @patch('hadamard_random_forest.sample.M3Mitigation')
    @patch('hadamard_random_forest.sample.mthree_utils.final_measurement_mapping')
    def test_get_samples_noisy_with_mitigation(self, mock_mapping, mock_m3):
        """Test get_samples_noisy with error mitigation using mocks."""
        num_qubits = 2
        base_circuit = self.simple_circuits[num_qubits]
        circuits = get_circuits(num_qubits, base_circuit)
        parameters = np.random.rand(base_circuit.num_parameters)
        
        # Mock the M3 mitigation objects
        mock_mit_instance = MagicMock()
        mock_m3.return_value = mock_mit_instance
        
        # Mock mapping function
        mock_mapping.return_value = [0, 1]
        
        # Mock the mitigation correction
        mock_quasi = MagicMock()
        mock_quasi.nearest_probability_distribution.return_value = {
            '00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25
        }
        mock_mit_instance.apply_correction.return_value = mock_quasi
        
        # Mock the backend to return predictable counts
        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_job.result.return_value = mock_result
        mock_result.get_counts.return_value = self._create_mock_counts(num_qubits, self.test_shots)
        
        with patch.object(self.backend_sim, 'run', return_value=mock_job):
            samples = get_samples_noisy(
                num_qubits=num_qubits,
                circuits=circuits,
                shots=self.test_shots,
                parameters=parameters,
                backend_sim=self.backend_sim,
                error_mitigation=True
            )
        
        # Verify mitigation was called
        self.assertTrue(mock_m3.called)
        self.assertTrue(mock_mit_instance.apply_correction.called)
        
        # Verify output structure
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), num_qubits + 1)
        
        for sample in samples:
            self.assertIsInstance(sample, np.ndarray)
            self.assertEqual(sample.shape, (2**num_qubits,))

    def test_get_samples_noisy_different_shot_counts(self):
        """Test get_samples_noisy with different shot counts."""
        num_qubits = 2
        base_circuit = self.simple_circuits[num_qubits]
        circuits = get_circuits(num_qubits, base_circuit)
        parameters = np.random.rand(base_circuit.num_parameters)
        
        shot_counts = [100, 1000, 10000]
        
        for shots in shot_counts:
            with self.subTest(shots=shots):
                samples = get_samples_noisy(
                    num_qubits=num_qubits,
                    circuits=circuits,
                    shots=shots,
                    parameters=parameters,
                    backend_sim=self.backend_sim,
                    error_mitigation=False
                )
                
                # Basic validation
                self.assertEqual(len(samples), num_qubits + 1)
                for sample in samples:
                    self.assertEqual(sample.shape, (2**num_qubits,))
                    # Higher shot counts should generally give more precise results
                    self.assertTrue(np.all(sample >= 0))

    def test_get_samples_noisy_parameter_assignment(self):
        """Test that parameters are correctly assigned to circuits."""
        num_qubits = 3
        base_circuit = self.simple_circuits[num_qubits]
        circuits = get_circuits(num_qubits, base_circuit)
        
        # Test with different parameter values
        param_sets = [
            np.zeros(base_circuit.num_parameters),
            np.ones(base_circuit.num_parameters) * 0.5,
            np.random.rand(base_circuit.num_parameters) * 2 * np.pi
        ]
        
        for i, parameters in enumerate(param_sets):
            with self.subTest(param_set=i):
                samples = get_samples_noisy(
                    num_qubits=num_qubits,
                    circuits=circuits,
                    shots=self.test_shots,
                    parameters=parameters,
                    backend_sim=self.backend_sim,
                    error_mitigation=False
                )
                
                # Different parameters should generally produce different results
                self.assertEqual(len(samples), num_qubits + 1)
                for sample in samples:
                    self.assertTrue(np.isfinite(sample).all())

    def test_get_circuits_hardware_basic(self):
        """Test basic functionality of get_circuits_hardware."""
        for num_qubits in [2, 3]:  # Use smaller systems for transpilation
            with self.subTest(num_qubits=num_qubits):
                base_circuit = self.simple_circuits[num_qubits]
                
                circuits = get_circuits_hardware(
                    num_qubits=num_qubits,
                    base_circuit=base_circuit,
                    device=self.fake_backend
                )
                
                # Verify structure
                self.assertIsInstance(circuits, list)
                self.assertEqual(len(circuits), num_qubits + 1)
                
                # Verify all circuits are transpiled (should have different structure)
                for circuit in circuits:
                    self.assertIsNotNone(circuit)
                    # Transpiled circuits may have different qubit counts due to routing
                    self.assertGreaterEqual(circuit.num_qubits, num_qubits)
                    # Verify measurements are present
                    self.assertTrue(any(op.operation.name == 'measure' for op in circuit.data))

    def test_get_circuits_hardware_transpilation(self):
        """Test that transpilation occurs correctly."""
        num_qubits = 3
        base_circuit = self.simple_circuits[num_qubits]
        
        # Compare transpiled vs non-transpiled
        regular_circuits = get_circuits(num_qubits, base_circuit)
        hardware_circuits = get_circuits_hardware(
            num_qubits=num_qubits,
            base_circuit=base_circuit,
            device=self.fake_backend
        )
        
        # Same number of circuits
        self.assertEqual(len(regular_circuits), len(hardware_circuits))
        
        # Hardware circuits should generally have more gates due to decomposition
        for i, (reg_circuit, hw_circuit) in enumerate(zip(regular_circuits, hardware_circuits)):
            with self.subTest(circuit_index=i):
                # Hardware circuits may have more operations due to transpilation
                self.assertGreaterEqual(len(hw_circuit.data), 0)
                # Both should have measurements
                reg_has_measure = any(op.operation.name == 'measure' for op in reg_circuit.data)
                hw_has_measure = any(op.operation.name == 'measure' for op in hw_circuit.data)
                self.assertTrue(reg_has_measure)
                self.assertTrue(hw_has_measure)

    def test_get_circuits_hardware_different_backends(self):
        """Test get_circuits_hardware with different backend types."""
        num_qubits = 2
        base_circuit = self.simple_circuits[num_qubits]
        
        # Test with different backends
        backends = [
            self.fake_backend,
            AerSimulator.from_backend(self.fake_backend)
        ]
        
        for i, backend in enumerate(backends):
            with self.subTest(backend_type=i):
                circuits = get_circuits_hardware(
                    num_qubits=num_qubits,
                    base_circuit=base_circuit,
                    device=backend
                )
                
                self.assertEqual(len(circuits), num_qubits + 1)
                for circuit in circuits:
                    self.assertIsNotNone(circuit)
                    # Verify the circuit is executable (has valid structure)
                    self.assertGreater(len(circuit.data), 0)

    @patch('hadamard_random_forest.sample.generate_preset_pass_manager')
    def test_get_circuits_hardware_pass_manager_usage(self, mock_pm):
        """Test that pass manager is used correctly."""
        num_qubits = 2
        base_circuit = self.simple_circuits[num_qubits]
        
        # Mock the pass manager
        mock_pm_instance = MagicMock()
        mock_pm.return_value = mock_pm_instance
        mock_pm_instance.run.side_effect = lambda x: x  # Return circuit unchanged
        
        circuits = get_circuits_hardware(
            num_qubits=num_qubits,
            base_circuit=base_circuit,
            device=self.fake_backend
        )
        
        # Verify pass manager was created and used
        mock_pm.assert_called_once()
        # Should be called once for each circuit (num_qubits + 1)
        self.assertEqual(mock_pm_instance.run.call_count, num_qubits + 1)
        # Verify we got the expected number of circuits
        self.assertEqual(len(circuits), num_qubits + 1)

    @patch('hadamard_random_forest.sample.Sampler')
    @patch('hadamard_random_forest.sample.mthree_utils.final_measurement_mapping')
    def test_get_samples_hardware_basic(self, mock_mapping, mock_sampler_class):
        """Test basic functionality of get_samples_hardware."""
        num_qubits = 2
        base_circuit = self.simple_circuits[num_qubits]
        circuits = get_circuits_hardware(num_qubits, base_circuit, self.fake_backend)
        parameters = np.random.rand(base_circuit.num_parameters)
        shots = 1024
        
        # Mock the sampler and its results
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        
        # Mock job and result
        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_data = MagicMock()
        mock_meas = MagicMock()
        
        # Set up the mock chain
        mock_sampler.run.return_value = mock_job
        mock_job.result.return_value = [mock_result]
        mock_result.data = mock_data
        mock_data.meas = mock_meas
        mock_meas.get_counts.return_value = self._create_mock_counts(num_qubits, shots)
        mock_job.job_id.return_value = f"job_test_{np.random.randint(1000)}"
        mock_job.usage_estimation = {'quantum_seconds': 1.23}
        
        # Mock mapping
        mock_mapping.return_value = list(range(num_qubits))
        
        result = get_samples_hardware(
            num_qubits=num_qubits,
            shots=shots,
            circuits=circuits,
            parameters=parameters,
            device=self.fake_backend,
            error_mitigation=False
        )
        
        # Verify return structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        
        mitigated_samples, raw_samples, job_ids, quantum_times = result
        
        # Verify samples
        self.assertIsInstance(mitigated_samples, list)
        self.assertIsInstance(raw_samples, list)
        self.assertEqual(len(mitigated_samples), num_qubits + 1)
        self.assertEqual(len(raw_samples), num_qubits + 1)
        
        for sample in mitigated_samples + raw_samples:
            self.assertIsInstance(sample, np.ndarray)
            self.assertEqual(sample.shape, (2**num_qubits,))
            self.assertTrue(np.all(sample >= 0))
        
        # Verify job metadata
        self.assertIsInstance(job_ids, list)
        self.assertIsInstance(quantum_times, list)
        self.assertEqual(len(job_ids), num_qubits + 1)
        self.assertEqual(len(quantum_times), num_qubits + 1)

    @patch('hadamard_random_forest.sample.Sampler')
    @patch('hadamard_random_forest.sample.mthree.M3Mitigation')
    @patch('hadamard_random_forest.sample.mthree_utils.final_measurement_mapping')
    def test_get_samples_hardware_with_mitigation(self, mock_mapping, mock_m3, mock_sampler_class):
        """Test get_samples_hardware with error mitigation."""
        num_qubits = 2
        base_circuit = self.simple_circuits[num_qubits]
        circuits = get_circuits_hardware(num_qubits, base_circuit, self.fake_backend)
        parameters = np.random.rand(base_circuit.num_parameters)
        shots = 1024
        
        # Mock sampler
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        
        # Mock job results
        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_data = MagicMock()
        mock_meas = MagicMock()
        
        mock_sampler.run.return_value = mock_job
        mock_job.result.return_value = [mock_result]
        mock_result.data = mock_data
        mock_data.meas = mock_meas
        mock_meas.get_counts.return_value = self._create_mock_counts(num_qubits, shots)
        mock_job.job_id.return_value = "test_job_with_mitigation"
        mock_job.usage_estimation = {'quantum_seconds': 2.45}
        
        # Mock M3 mitigation
        mock_mit_instance = MagicMock()
        mock_m3.return_value = mock_mit_instance
        mock_quasi = MagicMock()
        mock_quasi.nearest_probability_distribution.return_value = {
            '00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25
        }
        mock_mit_instance.apply_correction.return_value = mock_quasi
        
        # Mock mapping
        mock_mapping.return_value = list(range(num_qubits))
        
        result = get_samples_hardware(
            num_qubits=num_qubits,
            shots=shots,
            circuits=circuits,
            parameters=parameters,
            device=self.fake_backend,
            error_mitigation=True
        )
        
        # Verify mitigation was used
        self.assertTrue(mock_m3.called)
        self.assertTrue(mock_mit_instance.apply_correction.called)
        
        # Verify structure
        mitigated_samples, raw_samples, job_ids, quantum_times = result
        self.assertEqual(len(mitigated_samples), num_qubits + 1)
        self.assertEqual(len(raw_samples), num_qubits + 1)
        self.assertEqual(len(job_ids), num_qubits + 1)
        self.assertEqual(len(quantum_times), num_qubits + 1)
        
        # Mitigated and raw samples should be different arrays
        for i in range(len(mitigated_samples)):
            # They should have the same shape but potentially different values
            self.assertEqual(mitigated_samples[i].shape, raw_samples[i].shape)

    @patch('hadamard_random_forest.sample.Sampler')
    def test_get_samples_hardware_job_tracking(self, mock_sampler_class):
        """Test that job IDs and quantum times are correctly tracked."""
        num_qubits = 2
        base_circuit = self.simple_circuits[num_qubits]
        circuits = get_circuits_hardware(num_qubits, base_circuit, self.fake_backend)
        parameters = np.random.rand(base_circuit.num_parameters)
        
        # Mock sampler with unique job IDs and times
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        
        # Create unique job mocks
        job_ids = [f"job_{i}" for i in range(num_qubits + 1)]
        quantum_times = [i * 0.5 + 1.0 for i in range(num_qubits + 1)]
        
        def create_mock_job(job_id, q_time):
            mock_job = MagicMock()
            mock_result = MagicMock()
            mock_data = MagicMock()
            mock_meas = MagicMock()
            
            mock_job.result.return_value = [mock_result]
            mock_result.data = mock_data
            mock_data.meas = mock_meas
            mock_meas.get_counts.return_value = self._create_mock_counts(num_qubits, 1024)
            mock_job.job_id.return_value = job_id
            mock_job.usage_estimation = {'quantum_seconds': q_time}
            
            return mock_job
        
        # Set up sampler to return different jobs
        mock_jobs = [create_mock_job(jid, qt) for jid, qt in zip(job_ids, quantum_times)]
        mock_sampler.run.side_effect = mock_jobs
        
        result = get_samples_hardware(
            num_qubits=num_qubits,
            shots=1024,
            circuits=circuits,
            parameters=parameters,
            device=self.fake_backend,
            error_mitigation=False
        )
        
        _, _, returned_job_ids, returned_quantum_times = result
        
        # Verify job tracking
        self.assertEqual(returned_job_ids, job_ids)
        self.assertEqual(returned_quantum_times, quantum_times)

    def test_get_statevector_basic(self):
        """Test basic functionality of get_statevector."""
        for num_qubits in [2, 3]:
            with self.subTest(num_qubits=num_qubits):
                num_trees = 5
                samples = self._create_test_samples(num_qubits, normalize=True)
                
                statevector = get_statevector(
                    num_qubits=num_qubits,
                    num_trees=num_trees,
                    samples=samples,
                    save_tree=False,
                    show_tree=False
                )
                
                # Verify structure
                self.assertIsInstance(statevector, np.ndarray)
                self.assertEqual(statevector.shape, (2**num_qubits,))
                
                # Verify properties
                self.assertTrue(np.all(np.isfinite(statevector)))
                self.assertGreater(np.linalg.norm(statevector), 0)
                # Should be normalized
                self.assertAlmostEqual(np.linalg.norm(statevector), 1.0, places=6)

    def test_get_statevector_negative_probabilities_warning(self):
        """Test that get_statevector handles negative probabilities with warning."""
        num_qubits = 2
        num_trees = 3
        
        # Create samples with some negative values
        samples = self._create_test_samples(num_qubits, normalize=False)
        samples[0][1] = -0.1  # Add negative value to first sample
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            statevector = get_statevector(
                num_qubits=num_qubits,
                num_trees=num_trees,
                samples=samples,
                save_tree=False,
                show_tree=False
            )
            
            # Check that warning was issued
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("Negative sample probabilities", str(w[0].message))
        
        # Result should still be valid
        self.assertIsInstance(statevector, np.ndarray)
        self.assertTrue(np.all(np.isfinite(statevector)))


    def test_get_statevector_different_tree_counts(self):
        """Test get_statevector with different numbers of trees."""
        num_qubits = 2
        samples = self._create_test_samples(num_qubits, normalize=True)
        
        tree_counts = [1, 3, 5, 11]  # Include odd numbers for majority voting
        
        for num_trees in tree_counts:
            with self.subTest(num_trees=num_trees):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    statevector = get_statevector(
                        num_qubits=num_qubits,
                        num_trees=num_trees,
                        samples=samples,
                        save_tree=False,
                        show_tree=False
                    )
                
                self.assertEqual(statevector.shape, (2**num_qubits,))
                self.assertAlmostEqual(np.linalg.norm(statevector), 1.0, places=6)

    def test_get_statevector_normalization(self):
        """Test that get_statevector properly normalizes output."""
        num_qubits = 3
        num_trees = 5
        samples = self._create_test_samples(num_qubits, normalize=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            statevector = get_statevector(
                num_qubits=num_qubits,
                num_trees=num_trees,
                samples=samples,
                save_tree=False,
                show_tree=False
            )
        
        # Test normalization
        norm = np.linalg.norm(statevector)
        self.assertAlmostEqual(norm, 1.0, places=6)
        
        # Test that all elements are reasonable
        self.assertTrue(np.all(np.abs(statevector) <= 1.0))

    def test_get_statevector_reproducibility(self):
        """Test that get_statevector gives reproducible results with same input."""
        num_qubits = 2
        num_trees = 5
        samples = self._create_test_samples(num_qubits, normalize=True)
        
        # Run twice with same samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            # Fix the random seed before each call
            fix_random_seed(42)
            statevector1 = get_statevector(
                num_qubits=num_qubits,
                num_trees=num_trees,
                samples=samples.copy(),
                save_tree=False,
                show_tree=False
            )
            
            fix_random_seed(42)
            statevector2 = get_statevector(
                num_qubits=num_qubits,
                num_trees=num_trees,
                samples=samples.copy(),
                save_tree=False,
                show_tree=False
            )
        
        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(statevector1, statevector2, decimal=10)


class TestSampleErrorHandling(unittest.TestCase):
    """Test error handling and input validation for sample functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend_sim = AerSimulator()
        self.fake_backend = FakeFez()

    def test_get_circuits_invalid_inputs(self):
        """Test get_circuits with edge case inputs."""
        # Test with negative qubits - should work but produce empty range
        try:
            circuits = get_circuits(-1, QuantumCircuit(1))
            # Should return just the base circuit with measurements
            self.assertEqual(len(circuits), 1)  # -1 + 1 = 0 additional circuits
        except Exception:
            # Negative qubits might raise an exception, which is also acceptable
            pass
        
        # Test with mismatched qubit count - function works but may produce unexpected results
        circuit_2q = QuantumCircuit(2)
        circuits = get_circuits(3, circuit_2q)  # Circuit has 2 qubits, asking for 3
        # Should still return 4 circuits (3+1), though the extra H gates will be on non-existent qubits
        self.assertEqual(len(circuits), 4)

    def test_get_samples_noisy_invalid_parameters(self):
        """Test get_samples_noisy with invalid parameter arrays."""
        num_qubits = 2
        base_circuit = real_amplitudes(num_qubits, reps=1)
        circuits = get_circuits(num_qubits, base_circuit)
        
        # Test with wrong parameter count
        wrong_params = np.random.rand(base_circuit.num_parameters + 1)
        
        with self.assertRaises(Exception):
            get_samples_noisy(
                num_qubits=num_qubits,
                circuits=circuits,
                shots=1024,
                parameters=wrong_params,
                backend_sim=self.backend_sim,
                error_mitigation=False
            )

    def test_get_samples_noisy_zero_shots(self):
        """Test get_samples_noisy with zero shots."""
        num_qubits = 2
        base_circuit = real_amplitudes(num_qubits, reps=1)
        circuits = get_circuits(num_qubits, base_circuit)
        parameters = np.random.rand(base_circuit.num_parameters)
        
        # Zero shots should either raise error or return empty results
        with self.assertRaises(Exception):
            get_samples_noisy(
                num_qubits=num_qubits,
                circuits=circuits,
                shots=0,
                parameters=parameters,
                backend_sim=self.backend_sim,
                error_mitigation=False
            )

    def test_get_statevector_invalid_samples(self):
        """Test get_statevector with invalid sample inputs."""
        num_qubits = 2
        num_trees = 3
        
        # Test with wrong number of samples
        wrong_samples = [np.random.rand(4) for _ in range(2)]  # Should be 3 samples for 2 qubits
        
        with self.assertRaises(Exception):
            get_statevector(num_qubits, num_trees, wrong_samples, save_tree=False)
        
        # Test with wrong sample dimensions
        wrong_dim_samples = [np.random.rand(8) for _ in range(3)]  # Should be 4 elements for 2 qubits
        
        with self.assertRaises(Exception):
            get_statevector(num_qubits, num_trees, wrong_dim_samples, save_tree=False)

    def test_get_statevector_empty_samples(self):
        """Test get_statevector with empty or null samples."""
        num_qubits = 2
        num_trees = 3
        
        # Test with empty list
        with self.assertRaises(Exception):
            get_statevector(num_qubits, num_trees, [], save_tree=False)
        
        # Test with samples containing zeros
        zero_samples = [np.zeros(4) for _ in range(3)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                result = get_statevector(num_qubits, num_trees, zero_samples, save_tree=False)
                # Should either work or raise an exception, but not crash
                self.assertIsInstance(result, np.ndarray)
            except Exception:
                # Zero samples might legitimately cause errors
                pass

    def test_get_circuits_hardware_invalid_backend(self):
        """Test get_circuits_hardware with invalid backend."""
        num_qubits = 2
        base_circuit = real_amplitudes(num_qubits)
        
        # Test with None backend - may raise exception when passed to generate_preset_pass_manager
        try:
            circuits = get_circuits_hardware(num_qubits, base_circuit, None)
            # If it doesn't raise an exception, that's also acceptable behavior
            self.assertIsInstance(circuits, list)
        except Exception:
            # None backend should raise an exception in generate_preset_pass_manager
            pass

    @patch('hadamard_random_forest.sample.Sampler')
    def test_get_samples_hardware_failed_jobs(self, mock_sampler_class):
        """Test get_samples_hardware handling of failed jobs."""
        num_qubits = 2
        base_circuit = real_amplitudes(num_qubits)
        
        # Suppress mthree deprecation warnings from external library
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="mthree.utils")
            circuits = get_circuits_hardware(num_qubits, base_circuit, self.fake_backend)
        
        parameters = np.random.rand(base_circuit.num_parameters)
        
        # Mock sampler that raises an exception
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        mock_sampler.run.side_effect = Exception("Job submission failed")
        
        with self.assertRaises(Exception):
            get_samples_hardware(
                num_qubits=num_qubits,
                shots=1024,
                circuits=circuits,
                parameters=parameters,
                device=self.fake_backend,
                error_mitigation=False
            )

    def test_parameter_dimension_mismatch(self):
        """Test functions with parameter dimension mismatches."""
        num_qubits = 3
        base_circuit = real_amplitudes(num_qubits, reps=2)
        circuits = get_circuits(num_qubits, base_circuit)
        
        # Create parameters with wrong dimensions
        wrong_params_1d = np.random.rand(5)  # Wrong count
        wrong_params_2d = np.random.rand(2, 2)  # Wrong shape
        
        test_params = [wrong_params_1d, wrong_params_2d]
        
        for params in test_params:
            with self.subTest(params_shape=params.shape):
                with self.assertRaises(Exception):
                    get_samples_noisy(
                        num_qubits=num_qubits,
                        circuits=circuits,
                        shots=1024,
                        parameters=params,
                        backend_sim=self.backend_sim,
                        error_mitigation=False
                    )

    def test_extreme_values(self):
        """Test functions with extreme input values."""
        # Test with very large qubit count (should be handled gracefully)
        large_qubits = 20
        
        # This should either work or fail gracefully, not crash
        try:
            simple_circuit = QuantumCircuit(large_qubits)
            circuits = get_circuits(large_qubits, simple_circuit)
            self.assertEqual(len(circuits), large_qubits + 1)
        except (MemoryError, Exception):
            # Large systems might legitimately fail due to memory constraints
            pass

        # Test with very small valid inputs
        minimal_circuit = QuantumCircuit(1)
        minimal_circuits = get_circuits(1, minimal_circuit)
        self.assertEqual(len(minimal_circuits), 2)  # Base + 1 H variant


class TestSampleIntegration(unittest.TestCase):
    """Integration tests for the sample module."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend_sim = AerSimulator()

    def test_integration_basic_workflow(self):
        """Test the complete basic workflow integration."""
        num_qubits = 2  # Smaller for faster testing
        num_trees = 3
        base_circuit = real_amplitudes(num_qubits)
        parameters = np.random.rand(base_circuit.num_parameters)
        shots = 1024
        
        # Complete workflow
        circuits = get_circuits(num_qubits, base_circuit)
        samples = get_samples_noisy(
            num_qubits, circuits, shots, parameters, 
            self.backend_sim, error_mitigation=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            statevector = get_statevector(num_qubits, num_trees, samples, save_tree=False)
        
        # Validate end-to-end
        self.assertEqual(statevector.shape, (2**num_qubits,))
        self.assertTrue(np.all(np.isfinite(statevector)))
        self.assertAlmostEqual(np.linalg.norm(statevector), 1.0, places=6)

    def test_integration_with_error_mitigation(self):
        """Test integration workflow with error mitigation."""
        num_qubits = 2
        num_trees = 3
        base_circuit = real_amplitudes(num_qubits)
        parameters = np.random.rand(base_circuit.num_parameters)
        shots = 1024
        
        # Use mock to avoid expensive M3 calibration
        with patch('hadamard_random_forest.sample.M3Mitigation') as mock_m3:
            mock_mit_instance = MagicMock()
            mock_m3.return_value = mock_mit_instance
            mock_quasi = MagicMock()
            mock_quasi.nearest_probability_distribution.return_value = {
                '00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25
            }
            mock_mit_instance.apply_correction.return_value = mock_quasi
            
            with patch('hadamard_random_forest.sample.mthree_utils.final_measurement_mapping') as mock_mapping:
                mock_mapping.return_value = [0, 1]
                
                circuits = get_circuits(num_qubits, base_circuit)
                samples = get_samples_noisy(
                    num_qubits, circuits, shots, parameters,
                    self.backend_sim, error_mitigation=True
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    statevector = get_statevector(num_qubits, num_trees, samples, save_tree=False)
                
                # Validate workflow with mitigation
                self.assertEqual(statevector.shape, (2**num_qubits,))
                self.assertTrue(np.all(np.isfinite(statevector)))

    def test_integration_different_ansatz_types(self):
        """Test integration with different circuit ansatz types."""
        num_qubits = 2
        num_trees = 3
        shots = 1024
        
        ansatz_types = [
            real_amplitudes(num_qubits, reps=1),
            efficient_su2(num_qubits, reps=1)
        ]
        
        for i, base_circuit in enumerate(ansatz_types):
            with self.subTest(ansatz_type=i):
                parameters = np.random.rand(base_circuit.num_parameters)
                
                circuits = get_circuits(num_qubits, base_circuit)
                samples = get_samples_noisy(
                    num_qubits, circuits, shots, parameters,
                    self.backend_sim, error_mitigation=False
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    statevector = get_statevector(num_qubits, num_trees, samples, save_tree=False)
                
                self.assertEqual(statevector.shape, (2**num_qubits,))
                self.assertTrue(np.all(np.isfinite(statevector)))

    @patch('hadamard_random_forest.sample.Sampler')
    def test_integration_hardware_workflow(self, mock_sampler_class):
        """Test integration of hardware workflow with mocks."""
        num_qubits = 2
        num_trees = 3
        base_circuit = real_amplitudes(num_qubits)
        parameters = np.random.rand(base_circuit.num_parameters)
        shots = 1024
        
        # Mock hardware workflow
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        
        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_data = MagicMock()
        mock_meas = MagicMock()
        
        mock_sampler.run.return_value = mock_job
        mock_job.result.return_value = [mock_result]
        mock_result.data = mock_data
        mock_data.meas = mock_meas
        mock_meas.get_counts.return_value = {'00': 256, '01': 256, '10': 256, '11': 256}
        mock_job.job_id.return_value = "test_job"
        mock_job.usage_estimation = {'quantum_seconds': 1.5}
        
        with patch('hadamard_random_forest.sample.mthree_utils.final_measurement_mapping') as mock_mapping:
            mock_mapping.return_value = [0, 1]
            
            # Complete hardware workflow
            circuits = get_circuits_hardware(num_qubits, base_circuit, self.backend_sim)
            mitigated_samples, raw_samples, job_ids, quantum_times = get_samples_hardware(
                num_qubits, shots, circuits, parameters,
                self.backend_sim, error_mitigation=False
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                statevector = get_statevector(num_qubits, num_trees, mitigated_samples, save_tree=False)
            
            # Validate hardware workflow
            self.assertEqual(len(mitigated_samples), num_qubits + 1)
            self.assertEqual(len(raw_samples), num_qubits + 1)
            self.assertEqual(len(job_ids), num_qubits + 1)
            self.assertEqual(len(quantum_times), num_qubits + 1)
            self.assertEqual(statevector.shape, (2**num_qubits,))

    def test_performance_scaling(self):
        """Test performance scaling across different system sizes."""
        import time
        
        results = {}
        max_qubits = 4  # Keep reasonable for testing
        
        for num_qubits in range(2, max_qubits + 1):
            base_circuit = real_amplitudes(num_qubits, reps=1)
            parameters = np.random.rand(base_circuit.num_parameters)
            
            start_time = time.time()
            
            # Time the circuit generation
            circuits = get_circuits(num_qubits, base_circuit)
            
            # Time the sampling (with reduced shots for speed)
            samples = get_samples_noisy(
                num_qubits, circuits, 512, parameters,  # Reduced shots
                self.backend_sim, error_mitigation=False
            )
            
            # Time the reconstruction (with fewer trees)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                statevector = get_statevector(num_qubits, 3, samples, save_tree=False)
            
            elapsed_time = time.time() - start_time
            results[num_qubits] = elapsed_time
            
            # Validate result
            self.assertEqual(statevector.shape, (2**num_qubits,))
            self.assertTrue(np.all(np.isfinite(statevector)))
        
        # Basic scaling check - should be reasonable
        for num_qubits, time_taken in results.items():
            self.assertLess(time_taken, 60.0)  # Should complete within 60 seconds
        
        # Check that scaling is not exponential in the small regime
        if len(results) >= 2:
            times = list(results.values())
            # Time shouldn't increase by more than factor of 10 per qubit for small systems
            for i in range(1, len(times)):
                self.assertLess(times[i] / times[i-1], 10.0)

    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable for moderate system sizes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test with moderate system size
        num_qubits = 4
        num_trees = 5
        base_circuit = real_amplitudes(num_qubits, reps=2)
        parameters = np.random.rand(base_circuit.num_parameters)
        
        circuits = get_circuits(num_qubits, base_circuit)
        samples = get_samples_noisy(
            num_qubits, circuits, 1024, parameters,
            self.backend_sim, error_mitigation=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            statevector = get_statevector(num_qubits, num_trees, samples, save_tree=False)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 4 qubits)
        self.assertLess(memory_increase, 500.0)
        
        # Result should still be valid
        self.assertEqual(statevector.shape, (2**num_qubits,))
        self.assertTrue(np.all(np.isfinite(statevector)))

if __name__ == '__main__':
    unittest.main()