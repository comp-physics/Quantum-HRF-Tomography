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
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
import treelib
from math import comb  
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt


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



def generate_random_forest(
    num_qubits: int,
    num_trees: int,
    samples: List[np.ndarray],
    save_tree: bool = True,
    show_tree: bool = False,
    show_first: bool = False
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
            try:
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
            finally:
                # Always close the figure to prevent memory leaks
                plt.close()

        # Accumulate for majority voting
        if signs_stack is None:
            signs_stack = signs
        else:
            signs_stack = np.vstack([signs_stack, signs])

    assert signs_stack is not None
    return majority_voting(signs_stack)

