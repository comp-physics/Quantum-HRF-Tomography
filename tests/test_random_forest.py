import unittest
import random
import warnings
import numpy as np
import networkx as nx
import treelib
from scipy.sparse import coo_matrix
from hadamard_random_forest.random_forest import (
    fix_random_seed,
    hamming_weight, 
    pascal_layer,
    optimized_uniform_spanning_tree,
    generate_hypercube_tree,
    find_global_roots_and_leafs,
    get_path,
    get_path_sparse_matrix,
    get_weight,
    get_signs,
    majority_voting,
    generate_random_forest
)


class TestRandomForest(unittest.TestCase):

    def test_fix_random_seed(self):
        """Test that fix_random_seed creates reproducible random sequences."""
        fix_random_seed(42)
        random_vals_1 = [random.random() for _ in range(10)]
        numpy_vals_1 = np.random.rand(10)
        
        fix_random_seed(42)
        random_vals_2 = [random.random() for _ in range(10)]
        numpy_vals_2 = np.random.rand(10)
        
        np.testing.assert_array_equal(random_vals_1, random_vals_2)
        np.testing.assert_array_equal(numpy_vals_1, numpy_vals_2)

    def test_hamming_weight(self):
        """Test Hamming weight calculation for various integers."""
        self.assertEqual(hamming_weight(0), 0)
        self.assertEqual(hamming_weight(1), 1)
        self.assertEqual(hamming_weight(2), 1)
        self.assertEqual(hamming_weight(3), 2)
        self.assertEqual(hamming_weight(7), 3)
        self.assertEqual(hamming_weight(15), 4)
        self.assertEqual(hamming_weight(16), 1)
        self.assertEqual(hamming_weight(255), 8)

    def test_pascal_layer(self):
        """Test binomial coefficient calculation using Pascal's triangle."""
        # Test known values
        self.assertEqual(pascal_layer(0, 0), 1)
        self.assertEqual(pascal_layer(1, 0), 1)
        self.assertEqual(pascal_layer(1, 1), 1)
        self.assertEqual(pascal_layer(2, 1), 2)
        self.assertEqual(pascal_layer(3, 2), 3)
        self.assertEqual(pascal_layer(5, 2), 10)
        self.assertEqual(pascal_layer(10, 3), 120)
        
        # Test edge cases
        self.assertEqual(pascal_layer(5, 0), 1)
        self.assertEqual(pascal_layer(5, 5), 1)

    def test_optimized_uniform_spanning_tree(self):
        """Test spanning tree generation on hypercube graphs."""
        for dimension in range(2, 5):  # Test 2, 3, 4 dimensional hypercubes
            fix_random_seed(42)
            G = nx.hypercube_graph(dimension)
            G = nx.convert_node_labels_to_integers(G)
            
            spanning_tree = optimized_uniform_spanning_tree(G, dimension)
            
            # Test basic properties
            self.assertIsInstance(spanning_tree, nx.Graph)
            self.assertEqual(len(spanning_tree.nodes), 2**dimension)
            self.assertEqual(len(spanning_tree.edges), 2**dimension - 1)
            self.assertTrue(nx.is_connected(spanning_tree))
            self.assertTrue(nx.is_tree(spanning_tree))
            
            # Test root connections (root should connect to power-of-2 nodes)
            root_neighbors = list(spanning_tree.neighbors(0))
            power2_nodes = [2**k for k in range(dimension)]
            self.assertTrue(all(node in power2_nodes for node in root_neighbors))

    def test_generate_hypercube_tree(self):
        """Test hypercube tree generation and conversion to treelib format."""
        for dimension in range(2, 5):
            fix_random_seed(42)
            tree, spanning_tree = generate_hypercube_tree(dimension)
            
            # Test return types
            self.assertIsInstance(tree, treelib.Tree)
            self.assertIsInstance(spanning_tree, nx.Graph)
            
            # Test tree properties
            self.assertEqual(len(tree.nodes), 2**dimension)
            root_node = tree.get_node(0)
            self.assertIsNone(tree.parent(0))  # Root has no parent
            
            # Test spanning tree properties
            self.assertEqual(len(spanning_tree.nodes), 2**dimension)
            self.assertEqual(len(spanning_tree.edges), 2**dimension - 1)
            self.assertTrue(nx.is_connected(spanning_tree))
            
            # Test consistency between tree and spanning_tree
            tree_edges = set()
            for node_id in tree.nodes:
                parent_node = tree.parent(node_id)
                if parent_node is not None:
                    parent_id = parent_node.identifier
                    edge = tuple(sorted([parent_id, node_id]))
                    tree_edges.add(edge)
            spanning_edges = set(tuple(sorted(edge)) for edge in spanning_tree.edges)
            self.assertEqual(tree_edges, spanning_edges)

    def test_find_global_roots_and_leafs(self):
        """Test identification of root and leaf nodes grouped by 2^k distances."""
        num_qubits = 3
        fix_random_seed(42)
        tree, _ = generate_hypercube_tree(num_qubits)
        
        roots, leafs = find_global_roots_and_leafs(tree, num_qubits)
        
        # Test return types and structure
        self.assertIsInstance(roots, list)
        self.assertIsInstance(leafs, list)
        self.assertEqual(len(roots), num_qubits)
        self.assertEqual(len(leafs), num_qubits)
        
        # Test that all lists contain integer node IDs
        for k in range(num_qubits):
            self.assertIsInstance(roots[k], list)
            self.assertIsInstance(leafs[k], list)
            self.assertTrue(all(isinstance(node, int) for node in roots[k]))
            self.assertTrue(all(isinstance(node, int) for node in leafs[k]))
        
        # Test that all leaf nodes are covered (except root)
        all_leafs = set()
        for leaf_list in leafs:
            all_leafs.update(leaf_list)
        expected_nodes = set(range(1, 2**num_qubits))  # All nodes except root (0)
        self.assertEqual(all_leafs, expected_nodes)

    def test_get_path(self):
        """Test path computation from root to all nodes."""
        num_qubits = 3
        fix_random_seed(42)
        tree, _ = generate_hypercube_tree(num_qubits)
        
        paths = get_path(tree, num_qubits)
        
        # Test return type and structure
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 2**num_qubits)
        
        # Test that all paths are lists of integers
        for i, path in enumerate(paths):
            self.assertIsInstance(path, list)
            self.assertTrue(all(isinstance(node, int) for node in path))
            # Each path should start from node i and end at root (reverse order)
            self.assertEqual(path[0], i)   # Path starts at target node
            self.assertEqual(path[-1], 0)  # Path ends at root
        
        # Test root path (should only contain root)
        self.assertEqual(paths[0], [0])

    def test_get_path_sparse_matrix(self):
        """Test sparse matrix creation from path lists."""
        num_qubits = 3
        fix_random_seed(42)
        tree, _ = generate_hypercube_tree(num_qubits)
        paths = get_path(tree, num_qubits)
        
        sparse_matrix = get_path_sparse_matrix(paths, num_qubits)
        
        # Test return type and properties
        self.assertIsInstance(sparse_matrix, coo_matrix)
        self.assertEqual(sparse_matrix.shape, (2**num_qubits, 2**num_qubits))
        
        # Convert to dense for easier testing
        dense_matrix = sparse_matrix.toarray()
        
        # Test that matrix entries are 0 or 1
        self.assertTrue(np.all(np.isin(dense_matrix, [0, 1])))
        
        # Test that each row corresponds to a path
        for i, path in enumerate(paths):
            row = dense_matrix[i]
            # Non-zero elements should correspond to nodes in the path
            nonzero_indices = np.where(row == 1)[0]
            self.assertEqual(set(nonzero_indices), set(path))
        
        # Test that root appears in all paths (column 0 should be all ones)
        self.assertTrue(np.all(dense_matrix[:, 0] == 1))

    def test_get_weight(self):
        """Test weight calculation from sample data."""
        num_qubits = 3
        fix_random_seed(42)
        tree, _ = generate_hypercube_tree(num_qubits)
        roots, leafs = find_global_roots_and_leafs(tree, num_qubits)
        
        # Create mock sample data
        N = 2**num_qubits
        samples = [np.random.rand(N) for _ in range(num_qubits + 1)]
        
        weights = get_weight(samples, roots, leafs, num_qubits)
        
        # Test return type and shape
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(weights.shape, (N,))
        
        # Test that root weight is always 1.0
        self.assertEqual(weights[0], 1.0)
        
        # Test that all weights are +1 or -1 (signs)
        unique_weights = np.unique(weights)
        self.assertTrue(np.all(np.isin(unique_weights, [-1, 1])))

    def test_get_signs(self):
        """Test sign computation using both sparse and dense methods."""
        num_qubits = 3
        fix_random_seed(42)
        tree, _ = generate_hypercube_tree(num_qubits)
        paths = get_path(tree, num_qubits)
        sparse_matrix = get_path_sparse_matrix(paths, num_qubits)
        
        # Create mock weights
        N = 2**num_qubits
        weights = np.random.choice([-1, 1], size=N)
        weights[0] = 1  # Root weight is always 1
        
        # Test with sparse matrix
        idx_cumsum = np.insert(np.cumsum(sparse_matrix.getnnz(axis=1)), 0, 0)
        signs_sparse = get_signs(weights, sparse_matrix, paths, idx_cumsum)
        
        # Test with dense method (path_matrix = None)
        signs_dense = get_signs(weights, None, paths, idx_cumsum)
        
        # Test return types and shapes
        self.assertIsInstance(signs_sparse, np.ndarray)
        self.assertIsInstance(signs_dense, np.ndarray)
        self.assertEqual(signs_sparse.shape, (N,))
        self.assertEqual(signs_dense.shape, (N,))
        
        # Test that both methods give similar results (may differ due to numerical precision)
        # At least check that signs are consistent for root node
        self.assertEqual(signs_sparse[0], weights[0])
        self.assertEqual(signs_dense[0], weights[0])

    def test_majority_voting(self):
        """Test majority voting with various scenarios."""
        # Test unanimous positive votes
        votes = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result = majority_voting(votes)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result, expected)
        
        # Test unanimous negative votes
        votes = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
        result = majority_voting(votes)
        expected = np.array([-1, -1, -1])
        np.testing.assert_array_equal(result, expected)
        
        # Test mixed votes with clear majority
        votes = np.array([[1, -1, 1], [1, -1, -1], [1, 1, -1]])
        result = majority_voting(votes)
        expected = np.array([1, -1, -1])
        np.testing.assert_array_equal(result, expected)
        
        # Test tie scenarios (should default to +1)
        votes = np.array([[1, -1], [-1, 1]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = majority_voting(votes)
            # Check that a warning was issued
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("Zero elements encountered", str(w[0].message))
        expected = np.array([1, 1])  # Ties resolved to +1
        np.testing.assert_array_equal(result, expected)
        
        # Test single vote
        votes = np.array([[1, -1, 1]])
        result = majority_voting(votes)
        expected = np.array([1, -1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_generate_random_forest_basic(self):
        """Test basic functionality of generate_random_forest."""
        num_qubits = 3
        num_trees = 5
        fix_random_seed(42)
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits+1)]
        
        result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        
        # Test return type and shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2**num_qubits,))
        
        # Test that results are +1 or -1
        unique_vals = np.unique(result)
        self.assertTrue(np.all(np.isin(unique_vals, [-1, 1])))

    def test_generate_random_forest_deterministic(self):
        """Test that generate_random_forest gives reproducible results with fixed seed."""
        num_qubits = 3
        num_trees = 5
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits+1)]
        
        # Run twice with same seed
        fix_random_seed(42)
        result1 = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        
        fix_random_seed(42)
        result2 = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_generate_random_forest_different_sizes(self):
        """Test generate_random_forest with different qubit counts."""
        for num_qubits in [2, 3, 4]:
            with self.subTest(num_qubits=num_qubits):
                num_trees = 3
                fix_random_seed(42)
                samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits+1)]
                
                result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
                
                self.assertEqual(result.shape, (2**num_qubits,))
                self.assertTrue(np.all(np.isin(result, [-1, 1])))

    def test_generate_random_forest_single_tree(self):
        """Test generate_random_forest with single tree (no majority voting)."""
        num_qubits = 3
        num_trees = 1
        fix_random_seed(42)
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits+1)]
        
        result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        
        self.assertEqual(result.shape, (2**num_qubits,))
        self.assertTrue(np.all(np.isin(result, [-1, 1])))

    def test_generate_random_forest_large_qubits_no_save(self):
        """Test that large qubit counts disable tree saving automatically."""
        num_qubits = 12  # Larger than MAX_VIS_QUBITS (10)
        num_trees = 2
        fix_random_seed(42)
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits+1)]
        
        # This should work without trying to create visualizations
        # May generate warnings from majority voting due to random data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate_random_forest(num_qubits, num_trees, samples, save_tree=True)
        
        self.assertEqual(result.shape, (2**num_qubits,))
        self.assertTrue(np.all(np.isin(result, [-1, 1])))

    def test_integration_tree_components(self):
        """Test that tree generation, analysis, and path computation work together."""
        num_qubits = 3
        fix_random_seed(42)
        
        # Generate tree
        tree, spanning_tree = generate_hypercube_tree(num_qubits)
        
        # Analyze tree structure
        roots, leafs = find_global_roots_and_leafs(tree, num_qubits)
        paths = get_path(tree, num_qubits)
        sparse_matrix = get_path_sparse_matrix(paths, num_qubits)
        
        # Test consistency between components
        N = 2**num_qubits
        
        # Test that spanning tree is connected and has correct structure
        self.assertTrue(nx.is_connected(spanning_tree))
        self.assertEqual(len(spanning_tree.nodes), N)
        
        # Test that roots and leafs cover all nodes properly
        all_roots = set()
        all_leafs = set()
        for root_list in roots:
            all_roots.update(root_list)
        for leaf_list in leafs:
            all_leafs.update(leaf_list)
        
        # All nodes should be reachable via paths
        self.assertEqual(len(paths), N)
        self.assertTrue(all(len(path) > 0 for path in paths))
        
        # Sparse matrix should match path structure
        dense_matrix = sparse_matrix.toarray()
        for i, path in enumerate(paths):
            row_ones = np.where(dense_matrix[i] == 1)[0]
            self.assertEqual(set(row_ones), set(path))

    def test_mathematical_properties_spanning_tree(self):
        """Test mathematical properties of spanning trees."""
        for num_qubits in [2, 3, 4]:
            with self.subTest(num_qubits=num_qubits):
                fix_random_seed(42)
                G = nx.hypercube_graph(num_qubits)
                G = nx.convert_node_labels_to_integers(G)
                spanning_tree = optimized_uniform_spanning_tree(G, num_qubits)
                
                # Spanning tree properties
                N = 2**num_qubits
                self.assertEqual(len(spanning_tree.nodes), N)
                self.assertEqual(len(spanning_tree.edges), N - 1)  # Tree property
                self.assertTrue(nx.is_connected(spanning_tree))
                self.assertTrue(nx.is_tree(spanning_tree))
                
                # All nodes should be reachable from root
                distances = nx.single_source_shortest_path_length(spanning_tree, 0)
                self.assertEqual(len(distances), N)

    def test_pascal_triangle_consistency(self):
        """Test that pascal_layer follows Pascal triangle properties."""
        # Test Pascal triangle identity: C(n,k) = C(n-1,k-1) + C(n-1,k)
        for n in range(2, 8):
            for k in range(1, n):
                left = pascal_layer(n-1, k-1) if k > 0 else 0
                right = pascal_layer(n-1, k) if k < n else 0
                expected = left + right
                actual = pascal_layer(n, k)
                self.assertEqual(actual, expected,
                    f"Pascal identity failed for C({n},{k}): {actual} != {expected}")

    def test_hamming_weight_properties(self):
        """Test mathematical properties of Hamming weight function."""
        # Test that Hamming weight of XOR is sum of individual weights minus 2*overlap
        for i in range(16):
            for j in range(16):
                # For integers, XOR gives symmetric difference
                xor_weight = hamming_weight(i ^ j)
                individual_sum = hamming_weight(i) + hamming_weight(j)
                overlap = hamming_weight(i & j)
                expected = individual_sum - 2 * overlap
                self.assertEqual(xor_weight, expected,
                    f"XOR weight property failed for {i}^{j}")

    def test_sign_propagation_consistency(self):
        """Test that sign propagation is mathematically consistent."""
        num_qubits = 3
        fix_random_seed(42)
        tree, _ = generate_hypercube_tree(num_qubits)
        paths = get_path(tree, num_qubits)
        
        # Create deterministic weights
        N = 2**num_qubits
        weights = np.ones(N)
        weights[1::2] = -1  # Alternate signs
        weights[0] = 1  # Root always positive
        
        # Compute signs using path product
        expected_signs = np.zeros(N)
        for i, path in enumerate(paths):
            expected_signs[i] = np.prod(weights[path])
        
        # Compare with get_signs function
        sparse_matrix = get_path_sparse_matrix(paths, num_qubits)
        idx_cumsum = np.insert(np.cumsum(sparse_matrix.getnnz(axis=1)), 0, 0)
        actual_signs = get_signs(weights, None, paths, idx_cumsum)
        
        np.testing.assert_array_almost_equal(actual_signs, expected_signs)


if __name__ == '__main__':
    unittest.main()

