#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
"""

from __future__ import annotations
from typing import Dict, List, Tuple , Any

import numpy as np

import qiskit
import mthree
from qiskit.providers import Backend
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer.primitives import Sampler as Aer_Sampler
from mthree import M3Mitigation
import mthree.utils as mthree_utils

from .random_forest import generate_random_forest

def get_circuits(
    num_qubits: int,
    base_circuit: qiskit.QuantumCircuit
) -> List[qiskit.QuantumCircuit]:
    """
    Generate a list of circuits each with a single Hadamard on one qubit appended.

    Args:
        num_qubits: Total number of qubits.
        base_circuit: A QuantumCircuit to which measurements and H gates are appended.

    Returns:
        List of QuantumCircuit objects including the base circuit with measure_all
        and one variant with an H applied to each qubit.
    """
    circuits: List[qiskit.QuantumCircuit] = []
    # Base circuit with measurements
    circuits.append(base_circuit.measure_all(inplace=False))
    # Variants with extra Hadamard on each qubit
    for iq in range(num_qubits):
        qc = qiskit.QuantumCircuit(num_qubits)
        qc.compose(base_circuit, inplace=True)
        qc.h(iq)
        circuits.append(qc.measure_all(inplace=False))
    return circuits


def get_samples(
    num_qubits: int,
    sampler: Aer_Sampler | Sampler,
    circuits: List[qiskit.QuantumCircuit],
    parameters: np.ndarray
) -> List[np.ndarray]:
    """
    Execute circuits and collect probability distributions using a noiseless sampler.

    Args:
        num_qubits: Number of qubits (defines statevector size 2**num_qubits).
        sampler: Sampler object providing run().result().quasi_dists.
        circuits: List of QuantumCircuit to execute.
        parameters: 1D array of parameter values to bind to each circuit.

    Returns:
        List of 1D numpy arrays of length 2**num_qubits representing probabilities.
    """
    n = len(circuits)
    if isinstance(sampler, Aer_Sampler):
        results = sampler.run(circuits, [parameters] * n).result().quasi_dists
    elif isinstance(sampler, Sampler):
        results = sampler.run([(qc, parameters) for qc in circuits]).result()[0].data.meas.get_counts()
    else:
        raise ValueError("Sampler must be of type qiskit_aer.primitives.Sampler or qiskit_ibm_runtime.SamplerV2.")
    
    samples: List[np.ndarray] = []
    for res in results:
        proba = np.zeros(2**num_qubits, dtype=float)
        for idx, val in res.items():
            proba[idx] = val
        samples.append(proba)
    return samples


def get_samples_noisy(
    num_qubits: int,
    circuits: List[qiskit.QuantumCircuit],
    shots: int,
    parameters: np.ndarray,
    backend_sim: Backend,
    error_mitigation: bool = False
) -> List[np.ndarray]:
    """
    Transpile and run circuits with optional M3 error mitigation.

    Args:
        num_qubits: Number of qubits.
        circuits: List of QuantumCircuit to transpile and run.
        shots: Number of shots per circuit execution.
        parameters: Parameter values to assign.
        backend_sim: Qiskit backend to run circuits on.
        error_mitigation: If True, perform M3 calibration and mitigation.

    Returns:
        List of numpy arrays of length 2**num_qubits with (mitigated) probabilities.
    """

    # Generate a preset pass manager.
    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend_sim,
        layout_method="default",
        routing_method="sabre",
        seed_transpiler=999
    )
    samples: List[np.ndarray] = []

    if error_mitigation:
        # Dictionary to store unique mapping keys and their M3Mitigation objects.
        mapping_mit: Dict[str, M3Mitigation] = {}
        # Measurement results.
        counts_data: List[Tuple[Dict[str, int], Any, str]] = []

        # Transpile and run each circuit.
        for circuit in circuits:
            transpiled = pm.run(circuit)
            mapping = mthree_utils.final_measurement_mapping(transpiled)

            # Create a key for the mapping.
            key = str(mapping)

            # If this mapping hasn't been seen, calibrate a new mitigation object.
            if key not in mapping_mit:
                # print("=========== New M3 calibration detected ===========")
                mit = M3Mitigation(backend_sim)
                mit.cals_from_system(mapping)
                mapping_mit[key] = mit

            # Assign parameters and execute the circuit.
            transpiled.assign_parameters(parameters, inplace=True)
            counts = backend_sim.run(transpiled, shots=shots).result().get_counts()
            counts_data.append((counts, mapping, key))

        # Apply error mitigation to each result.
        for counts, mapping, key in counts_data:
            mit = mapping_mit[key]
            # print(f"Applying M3 error mitigation with mapping: {mapping}")
            quasi = mit.apply_correction(counts, mapping)

            # Convert counts to a probability distribution.
            probs = quasi.nearest_probability_distribution()
            dist = {k: v / shots for k, v in qiskit.result.ProbDistribution(probs, shots=shots).items()}

            # Build a probability vector.
            proba = np.zeros(2**num_qubits, dtype=float)
            for idx, val in dist.items():
                proba[idx] = val
            samples.append(proba)
    else:
        for circuit in circuits:
            transpiled = pm.run(circuit)
            transpiled.assign_parameters(parameters, inplace=True)
            counts = backend_sim.run(transpiled, shots=shots).result().get_counts()
            proba = np.zeros(2**num_qubits, dtype=float)
            for bitstr, count in counts.items():
                idx = int(bitstr, 2)
                proba[idx] = count / shots
            samples.append(proba)
    return samples


def get_circuits_hardware(
    num_qubits: int,
    base_circuit: qiskit.QuantumCircuit,
    device: Backend
) -> List[qiskit.QuantumCircuit]:
    """
    Transpile a base circuit for hardware and generate variants with an appended Hadamard gate.

    Args:
        num_qubits: Total number of qubits.
        base_circuit: The original QuantumCircuit to transpile and append to.
        device: Qiskit backend or simulator to target for transpilation.

    Returns:
        A list of transpiled QuantumCircuit objects:
          - The first is the base circuit with measurements.
          - Each subsequent circuit has an additional H gate on qubit i before measurement.
    """
    # Create a pass manager for transpilation
    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=device,
        layout_method="default",
        routing_method="sabre",
        seed_transpiler=999
    )

    circuits: List[qiskit.QuantumCircuit] = []
    # Base circuit: add measurements and transpile
    qc_base = base_circuit.measure_all(inplace=False)
    circuits.append(pm.run(qc_base))

    # Variants: apply Hadamard on each qubit, then measure and transpile
    for qubit in range(num_qubits):
        qc = qiskit.QuantumCircuit(num_qubits)
        qc.compose(base_circuit, inplace=True)
        qc.h(qubit)
        qc.measure_all(inplace=True)
        circuits.append(pm.run(qc))

    return circuits


def get_samples_hardware(
    num_qubits: int,
    shots: int,
    circuits: List[qiskit.QuantumCircuit],
    parameters: np.ndarray,
    device: Backend,
    error_mitigation: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[float]]:
    """
    Execute circuits on hardware with optional M3 error mitigation and record raw and mitigated samples.

    Args:
        num_qubits: Number of qubits (defines vector size 2**num_qubits).
        shots: Number of shots per circuit execution.
        circuits: List of transpiled QuantumCircuit objects.
        parameters: 1D array of parameter values to bind to each circuit.
        device: Qiskit backend to run circuits on.
        error_mitigation: If True, perform M3 calibration and apply measurement mitigation.

    Returns:
        A tuple of four items:
            mitigated_samples: List of numpy arrays (length 2**num_qubits) after mitigation.
            raw_samples:       List of numpy arrays without mitigation.
            job_ids:           List of job ID strings for each circuit execution.
            quantum_times:     List of quantum execution times (in seconds).
    """
    # Prepare sampler for hardware
    sampler = Sampler(device)
    sampler.options.default_shots = shots

    mapping_mit: dict = {}
    results = []  # List of tuples: (counts, mapping_key)
    job_ids: List[str] = []
    quantum_times: List[float] = []

    # Submit jobs and collect raw counts
    for idx, circ in enumerate(circuits):
        # Measurement mitigation setup
        mapping = mthree_utils.final_measurement_mapping(circ)
        key = str(mapping)
        if error_mitigation and key not in mapping_mit:
            # print("=========== New M3 calibration detected ===========")
            mit = mthree.M3Mitigation(device)
            mit.cals_from_system(mapping)
            mapping_mit[key] = mit

        # Run circuit on hardware
        job = sampler.run([(circ, parameters)])
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        results.append((counts, key))

        job_ids.append(job.job_id())
        quantum_times.append(job.usage_estimation.get('quantum_seconds', 0.0))

    # Process raw samples
    raw_samples: List[np.ndarray] = []
    for counts, _ in results:
        vec = np.zeros(2**num_qubits, dtype=float)
        for bitstr, cnt in counts.items():
            idx = int(bitstr, 2)
            vec[idx] = cnt / shots
        raw_samples.append(vec)

    # Apply mitigation if requested
    mitigated_samples: List[np.ndarray] = []
    for (counts, key), raw in zip(results, raw_samples):
        if error_mitigation:
            mit = mapping_mit[key]
            quasi = mit.apply_correction(counts, mthree_utils.final_measurement_mapping(circuits[0]))
            probs = quasi.nearest_probability_distribution()
            vec = np.zeros(2**num_qubits, dtype=float)
            for bitstr, p in probs.items():
                vec[int(bitstr, 2)] = p
            mitigated_samples.append(vec)
        else:
            mitigated_samples.append(raw.copy())

    return mitigated_samples, raw_samples, job_ids, quantum_times

def get_statevector(
    num_qubits: int,
    num_trees: int,
    samples: List[np.ndarray],
    save_tree: bool = True,
    show_tree: bool = False 
) -> np.ndarray:
    """
    Construct the estimated statevector from measured samples and sign forest.
    Allows passing save_tree flag to control tree visualization.

    Args:
        num_qubits: Cube dimension (log2 of state size).
        num_trees: Number of trees in the random forest.
        samples: List of sample probability arrays.
        save_tree: If True, save the first 10 forest tree visualizations.

    Returns:
        A 1D numpy array of length 2**num_qubits representing the statevector.
    """
    # Compute amplitudes
    base = samples[0]
    if np.any(base < 0):
        import warnings
        warnings.warn("Negative sample probabilities found; using absolute values.")
        amplitudes = np.sqrt(np.abs(base))
    else:
        amplitudes = np.sqrt(base)

    # Generate signs (with optional save_tree)
    signs = generate_random_forest(
        num_qubits=num_qubits,
        num_trees=num_trees,
        samples=samples,
        save_tree=save_tree,
        show_tree=show_tree
    )

    # Normalization
    statevector = amplitudes * signs
    statevector = statevector/np.linalg.norm(statevector)

    return statevector