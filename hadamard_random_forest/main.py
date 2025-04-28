#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
"""

from __future__ import annotations
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple 

import numpy as np
import networkx as nx
import treelib
from math import comb  
from scipy import sparse  
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

import qiskit
import mthree
from qiskit.providers import Backend
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_aer.primitives import Sampler as Aer_Sampler
from mthree import M3Mitigation
import mthree.utils as mthree_utils

# Public API
__all__ = [
    "fix_random_seed",
    "hamming_weight",
    "pascal_layer",
    "optimized_uniform_spanning_tree",
    "generate_hypercube_tree",
    "find_global_roots_and_leafs",
    "get_path",
    "get_path_sparse_matrix",
    "get_weight",
    "get_signs",
    "majority_voting",
    "get_circuits",
    "get_samples",
    "get_samples_noisy",
    "get_circuits_hardware",
    "get_samples_hardware",
    "generate_random_forest",
    "get_statevector",
]

def fix_random_seed(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def hamming_weight(x: int) -> int:
    """
    Compute the Hamming weight (number of 1 bits) of an integer.

    Args:
        x: Integer whose binary representation is to be counted.

    Returns:
        The Hamming weight of `x`.
    """
    return bin(x).count("1")


def pascal_layer(n: int, k: int) -> int:
    """
    Compute the binomial coefficient C(n, k) using Pascal's triangle.

    Args:
        n: Total number of items.
        k: Number of items chosen (depth in Pascal's triangle).

    Returns:
        The binomial coefficient C(n, k).
    """
    return comb(n, k, exact=True)


def optimized_uniform_spanning_tree(G: nx.Graph, dimension: int) -> nx.Graph:
    """
    Generate a structured Uniform Spanning Tree (UST) on an n-dimensional hypercube graph.

    This method ensures:
      1. Node 0 connects to all power-of-2 nodes at depth 1.
      2. Subsequent layers follow Pascal triangle structure via a BFS-like expansion.

    Args:
        G: The hypercube graph (NetworkX Graph) with integer labels 0,...,2^n-1.
        dimension: Dimension n of the hypercube (number of qubits).

    Returns:
        A NetworkX Graph representing the spanning tree.
    """
    tree = nx.Graph()
    root = 0
    tree.add_node(root)

    # Connect root to all power-of-2 nodes (depth 1)
    power2_nodes = [node for node in G.nodes if node > 0 and (node & (node - 1)) == 0]
    first_depth = sorted(set(G.neighbors(root)) & set(power2_nodes))
    for node in first_depth:
        tree.add_edge(root, node)

    # Build layers according to Hamming weight
    layers = {k: [node for node in G.nodes if hamming_weight(node) == k] for k in range(dimension + 1)}
    available = set(first_depth) | {root}

    # BFS-like expansion for layers 2,...,n
    for k in range(2, dimension + 1):
        for node in layers[k]:
            parents = list(set(G.neighbors(node)) & available)
            if parents:
                parent = random.choice(parents)
                tree.add_edge(parent, node)
                available.add(node)
    return tree


def generate_hypercube_tree(dimension: int) -> Tuple[treelib.Tree, nx.Graph]:
    """
    Generate a spanning tree from an n-dimensional hypercube graph and convert it to a treelib.Tree.

    Args:
        dimension: Dimension of the hypercube (equal to num_qubits).

    Returns:
        A tuple (tree, spanning_tree) where:
            tree: A treelib.Tree object representing the hierarchy of nodes.
            spanning_tree: The NetworkX Graph of the generated spanning tree.
    """
    # Build and label hypercube graph
    G = nx.hypercube_graph(dimension)
    G = nx.convert_node_labels_to_integers(G)

    # Generate optimized UST
    spanning_tree = optimized_uniform_spanning_tree(G, dimension)

    # Convert to treelib.Tree
    tree = treelib.Tree()
    root = 0
    tree.create_node(str(root), root)

    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in spanning_tree.neighbors(node):
            if nbr not in visited:
                tree.create_node(str(nbr), nbr, parent=node)
                visited.add(nbr)
                queue.append(nbr)

    return tree, spanning_tree



def find_global_roots_and_leafs(tree: treelib.Tree, num_qubits: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Identify global root nodes and leaf nodes grouped by 2^k distances.

    Args:
        tree: A treelib.Tree representing the hierarchical structure of nodes.
        num_qubits: Number of qubits (log2 of total node count).

    Returns:
        roots: A list where each sublist contains root node IDs at distance 2^k.
        leafs: A list where each sublist contains leaf node IDs at distance 2^k.
    """
    nodes = list(tree.nodes.keys())
    roots: List[List[int]] = [[0] for _ in range(num_qubits)]
    leafs: List[List[int]] = [[2**k] for k in range(num_qubits)]

    for k in range(num_qubits):
        step = 2**k
        for node in nodes:
            for child in tree.children(node):
                if child.identifier - node == step:
                    roots[k].append(node)
                    leafs[k].append(child.identifier)
        roots[k] = sorted(set(roots[k]))
        leafs[k] = sorted(set(leafs[k]))

    # Validate completeness
    unique_leaves = set(sum(leafs, []))
    expected = set(range(1, 2**num_qubits))
    if unique_leaves != expected:
        logging.warning("Incomplete roots/leafs detection; missing %s", expected - unique_leaves)

    return roots, leafs


def get_path(tree: treelib.Tree, num_qubits: int) -> List[List[int]]:
    """
    Create paths from the root to all nodes in the tree.

    Args:
        tree: A treelib.Tree of nodes.
        num_qubits: Number of qubits (for total node count).

    Returns:
        A list where each element is the path (list of node IDs) from the root to node i.
    """
    total = 2**num_qubits
    paths: List[List[int]] = []
    for node_id in range(total):
        paths.append(list(tree.rsearch(node_id)))
    return paths


def get_path_sparse_matrix(path_to_node: List[List[int]], num_qubits: int) -> coo_matrix:
    """
    Transform path lists into a sparse COO matrix representation.

    Args:
        path_to_node: List of paths, where path_to_node[i] is the list of node IDs on the path to node i.
        num_qubits: Number of qubits (for matrix dimensions).

    Returns:
        A scipy.sparse.coo_matrix of shape (N, N) with ones indicating path membership.
    """
    N = 2**num_qubits
    row_idx: List[int] = []
    col_idx: List[int] = []
    data: List[int] = []
    for i, path in enumerate(path_to_node):
        row_idx.extend([i] * len(path))
        col_idx.extend(path)
        data.extend([1] * len(path))
    return coo_matrix((data, (row_idx, col_idx)), shape=(N, N))


def get_weight(
    samples: List[np.ndarray],
    global_root: List[List[int]],
    global_leaf: List[List[int]],
    num_qubits: int
) -> np.ndarray:
    """
    Compute relative weights/signs between parent and child nodes from sample data.

    Args:
        samples: List of sample probability arrays (one per circuit evaluation).
        global_root: Root node indices grouped by 2^k distances.
        global_leaf: Leaf node indices grouped by 2^k distances.
        num_qubits: Number of qubits (log2 of total node count).

    Returns:
        A numpy array of relative weights with shape (2**num_qubits,).
    """
    N = 2**num_qubits
    weights = np.zeros(N, dtype=float)
    weights[0] = 1.0

    for k in range(num_qubits):
        roots = global_root[k]
        leafs = global_leaf[k]
        signs = np.sign(
            2 * samples[k + 1][roots]
            - samples[0][roots]
            - samples[0][leafs]
        )
        weights[leafs] = signs

    return weights


def get_signs(
    weights: np.ndarray,
    path_matrix: Optional[coo_matrix],
    path_to_node: List[List[int]],
    idx_path_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute overall sign for each node based on weights along its path.

    Args:
        weights: Array of relative weights for each node.
        path_matrix: Optional sparse path matrix (COO) of shape (N, N).
        path_to_node: List of paths for each node.
        idx_path_matrix: 1D array of indices for reduceat on path_matrix.data.

    Returns:
        A numpy array of computed signs for each node.
    """
    if path_matrix is None:
        signs = np.zeros_like(weights)
        for i, path in enumerate(path_to_node):
            signs[i] = np.prod(weights[path], axis=0)
        return signs

    # Using sparse matrix reduction
    data = path_matrix.multiply(weights).data
    return np.multiply.reduceat(data, idx_path_matrix[:-1])


def majority_voting(votes: np.ndarray) -> np.ndarray:
    """
    Perform majority voting on binary (+1/-1) vote vectors.

    Args:
        votes: Array of shape (m, N) where m is number of vote vectors and
               N is number of elements in each vector, entries must be +1 or -1.

    Returns:
        A 1D array of length N with majority-voted values (+1 or -1) as the final sign determination result.
    """
    vote_sum = np.sum(votes, axis=0)
    result = np.atleast_1d(np.sign(vote_sum))
    if np.any(result == 0):
        warnings.warn(
            "Zero elements encountered in majority vote; replacing zeros with +1.",
            UserWarning
        )
        result[result == 0] = 1
    return result


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
    results = sampler.run(circuits, [parameters] * n).result().quasi_dists
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


def generate_random_forest(
    num_qubits: int,
    num_trees: int,
    samples: List[np.ndarray],
    save_tree: bool = True,
    show_tree: bool = False
) -> np.ndarray:
    """
    Build multiple random spanning trees on a hypercube and aggregate signs by majority voting.
    Optionally save visualizations of the first 10 trees in a structured folder.

    Args:
        num_qubits: Cube dimension (log2 of state size).
        num_trees: Number of random trees to generate.
        samples: List of sample probability arrays used to compute weights.
        save_tree: If True, save the first 5 tree plots under 'forest gallery/{num_qubits}-qubit/'.

    Returns:
        A 1D numpy array of length 2**num_qubits containing final +1/-1 signs.
    """
    MAX_VIS_QUBITS = 10
    if num_qubits > MAX_VIS_QUBITS:
        # Disable any tree saving or showing beyond the threshold
        if save_tree or show_first:
            logging.warning(
                "Too large to render the tree graph for num_qubits > %d (got %d).",
                MAX_VIS_QUBITS, num_qubits
            )
        save_tree = False
        show_first = False


    signs_stack: Optional[np.ndarray] = None

    # Prepare output directory if needed
    if save_tree:
        base_dir = Path("forest gallery") / f"{num_qubits}-qubit"
        base_dir.mkdir(parents=True, exist_ok=True)

    for m in range(num_trees):
        # Step 1: generate random spanning tree
        tree, spanning = generate_hypercube_tree(num_qubits)

        # Step 2: find global roots and leaves
        roots, leafs = find_global_roots_and_leafs(tree, num_qubits)

        # Step 3: convert to matrix form for parallel sign computation
        paths = get_path(tree, num_qubits)
        pmatrix = get_path_sparse_matrix(paths, num_qubits)
        idx_cumsum = np.insert(np.cumsum(pmatrix.getnnz(axis=1)), 0, 0)

        # Step 4: compute weights and signs
        weights = get_weight(samples, roots, leafs, num_qubits)
        signs = get_signs(weights, pmatrix, paths, idx_cumsum)

        # Optional: save first 5 tree visualizations
        if save_tree and m < 5:
            G = nx.hypercube_graph(num_qubits)
            G = nx.convert_node_labels_to_integers(G)
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

            # Dynamically size the figure
            base_size = 6
            extra = max(0, num_qubits - 5)
            width_factor  = 2 ** extra
            height_factor = 1.5 ** extra
            plt.figure(figsize=(base_size * width_factor, base_size * height_factor))

            nx.draw_networkx_edges(G, pos, edge_color='tab:gray', alpha=0.2, width=2)
            nx.draw_networkx_edges(spanning, pos, edge_color='tab:gray', width=3)
            node_colors = ['tab:blue' if s == 1 else 'tab:orange' for s in signs]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, edgecolors='black')
            nx.draw_networkx_labels(G, pos, font_color="white")

            plt.axis('off')
            plt.tight_layout()
            fig_path = base_dir / f"tree_{m}.png"
            plt.savefig(fig_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=200)

            if show_tree and m == 0:
                # this will pop up the first tree in-line (or in a window)
                plt.show()

            plt.close()

        # Accumulate for majority voting
        if signs_stack is None:
            signs_stack = signs
        else:
            signs_stack = np.vstack([signs_stack, signs])

    assert signs_stack is not None
    return majority_voting(signs_stack)

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