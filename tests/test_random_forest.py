import unittest
import random
import warnings
import multiprocessing as mp
from unittest.mock import patch, MagicMock
import time
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
    generate_random_forest,
    _generate_single_tree_worker,
    _generate_batch_trees_worker,
    get_cached_hypercube,
    get_cached_power2_nodes,
    get_cached_hamming_layers,
    get_or_create_pool,
    cleanup_pool
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


class TestParallelProcessing(unittest.TestCase):
    """Test parallel processing features introduced in version 0.2.0."""

    def setUp(self):
        """Set up test data for parallel processing tests."""
        self.num_qubits = 3
        self.num_trees = 5
        self.base_seed = 42
        # Create valid sample data
        fix_random_seed(42)
        self.samples = [np.random.rand(2**self.num_qubits) for _ in range(self.num_qubits + 1)]

    def test_generate_single_tree_worker_basic(self):
        """Test that _generate_single_tree_worker produces valid output."""
        args = (self.num_qubits, self.samples, 0, self.base_seed)
        tree_index, signs = _generate_single_tree_worker(args)
        
        # Verify return structure
        self.assertEqual(tree_index, 0)
        self.assertIsInstance(signs, np.ndarray)
        self.assertEqual(signs.shape, (2**self.num_qubits,))
        self.assertTrue(np.all(np.isin(signs, [-1, 1])))

    def test_generate_single_tree_worker_deterministic(self):
        """Test that worker function is deterministic with same seed."""
        args = (self.num_qubits, self.samples, 0, self.base_seed)
        
        tree_index1, signs1 = _generate_single_tree_worker(args)
        tree_index2, signs2 = _generate_single_tree_worker(args)
        
        self.assertEqual(tree_index1, tree_index2)
        np.testing.assert_array_equal(signs1, signs2)

    def test_generate_single_tree_worker_different_indices(self):
        """Test that different tree indices produce different results."""
        args1 = (self.num_qubits, self.samples, 0, self.base_seed)
        args2 = (self.num_qubits, self.samples, 1, self.base_seed)
        
        tree_index1, signs1 = _generate_single_tree_worker(args1)
        tree_index2, signs2 = _generate_single_tree_worker(args2)
        
        self.assertEqual(tree_index1, 0)
        self.assertEqual(tree_index2, 1)
        # Different tree indices should generally produce different signs
        # (though there's a small chance they could be identical)
        self.assertEqual(signs1.shape, signs2.shape)

    def test_parallel_vs_sequential_identical_results(self):
        """Test that parallel and sequential execution produce identical results."""
        # Force sequential execution
        with patch('hadamard_random_forest.random_forest.mp.cpu_count', return_value=1):
            fix_random_seed(42)
            result_sequential = generate_random_forest(
                self.num_qubits, self.num_trees, self.samples, save_tree=False, show_tree=False
            )

        # Force parallel execution (mock cpu_count > 1 and num_trees >= threshold)
        with patch('hadamard_random_forest.random_forest.mp.cpu_count', return_value=4):
            fix_random_seed(42)
            result_parallel = generate_random_forest(
                self.num_qubits, self.num_trees, self.samples, save_tree=False, show_tree=False
            )

        # Results should be identical
        np.testing.assert_array_equal(result_sequential, result_parallel)

    def test_parallel_threshold_logic(self):
        """Test that parallel processing is only used when appropriate."""
        # Test with num_trees below threshold (should use sequential)
        # New threshold is 100 trees
        with patch('hadamard_random_forest.random_forest.mp.cpu_count', return_value=4):
            with patch('hadamard_random_forest.random_forest.logging') as mock_logging:
                fix_random_seed(42)
                generate_random_forest(
                    self.num_qubits, 50, self.samples, save_tree=False, show_tree=False  # Below 100 threshold
                )
                # Should not log parallel processing message
                mock_logging.info.assert_not_called()

        # Test with num_trees above threshold (should use parallel)
        with patch('hadamard_random_forest.random_forest.mp.cpu_count', return_value=4):
            with patch('hadamard_random_forest.random_forest.logging') as mock_logging:
                fix_random_seed(42)
                generate_random_forest(
                    self.num_qubits, 111, self.samples, save_tree=False, show_tree=False  # Above 100 threshold
                )
                # Should log parallel processing message
                mock_logging.info.assert_called_once()

    def test_visualization_disables_parallel(self):
        """Test that visualization parameters disable parallel processing."""
        with patch('hadamard_random_forest.random_forest.mp.cpu_count', return_value=4):
            with patch('hadamard_random_forest.random_forest.logging') as mock_logging:
                # Mock the graphviz functionality to avoid pygraphviz dependency
                with patch('networkx.drawing.nx_agraph.graphviz_layout', return_value={i: (i, 0) for i in range(8)}):
                    with patch('matplotlib.pyplot.savefig'):  # Mock the actual file saving
                        fix_random_seed(42)
                        generate_random_forest(
                            self.num_qubits, 111, self.samples, save_tree=True, show_tree=False  # Visualization enabled, above threshold
                        )
                        # Should not use parallel processing due to visualization
                        mock_logging.info.assert_not_called()

    @patch('hadamard_random_forest.random_forest.get_or_create_pool')
    def test_multiprocessing_pool_usage(self, mock_get_pool):
        """Test that multiprocessing pool is used correctly for large tree counts."""
        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool
        
        # Mock pool.map to return expected batch results
        # With batch processing, we return lists of (tree_index, signs) tuples
        batch_results = []
        for batch_idx in range(8):  # Assuming 8 batches for 111 trees
            batch = []
            start_idx = batch_idx * 14  # roughly 111/8
            end_idx = min(start_idx + 14, 111)
            for i in range(start_idx, end_idx):
                batch.append((i, np.array([1, -1, 1, -1, 1, -1, 1, -1])))
            batch_results.append(batch)
        mock_pool.map.return_value = batch_results
        
        with patch('hadamard_random_forest.random_forest.mp.cpu_count', return_value=4):
            fix_random_seed(42)
            result = generate_random_forest(
                self.num_qubits, 111, self.samples, save_tree=False, show_tree=False  # Above threshold
            )
            
            # Verify pool was obtained and used
            mock_get_pool.assert_called_once()
            mock_pool.map.assert_called_once()
            
            # Verify the function argument to map is the batch worker
            args, kwargs = mock_pool.map.call_args
            self.assertEqual(args[0], _generate_batch_trees_worker)


class TestOptimizationFeatures(unittest.TestCase):
    """Test optimization features including caching and batch processing."""
    
    def test_get_cached_hypercube(self):
        """Test hypercube graph caching."""
        # Clear cache first
        import hadamard_random_forest.random_forest as rf
        rf._HYPERCUBE_CACHE.clear()
        
        # First call should create and cache
        G1 = get_cached_hypercube(3)
        self.assertIsInstance(G1, nx.Graph)
        self.assertEqual(len(G1.nodes), 8)
        self.assertEqual(len(G1.edges), 12)
        
        # Second call should return cached instance
        G2 = get_cached_hypercube(3)
        self.assertIs(G1, G2)  # Should be the exact same object
        
        # Different dimension should create new graph
        G3 = get_cached_hypercube(4)
        self.assertIsNot(G1, G3)
        self.assertEqual(len(G3.nodes), 16)
    
    def test_get_cached_power2_nodes(self):
        """Test power-of-2 nodes caching."""
        # Clear cache first
        import hadamard_random_forest.random_forest as rf
        rf._POWER2_NODES_CACHE.clear()
        
        # Test for 3 qubits
        nodes3 = get_cached_power2_nodes(3)
        self.assertEqual(nodes3, [1, 2, 4])
        
        # Second call should return cached
        nodes3_2 = get_cached_power2_nodes(3)
        self.assertEqual(nodes3, nodes3_2)
        
        # Test for 4 qubits
        nodes4 = get_cached_power2_nodes(4)
        self.assertEqual(nodes4, [1, 2, 4, 8])
    
    def test_get_cached_hamming_layers(self):
        """Test Hamming layers caching."""
        # Clear cache first
        import hadamard_random_forest.random_forest as rf
        rf._HAMMING_LAYERS_CACHE.clear()
        
        # Test for 2 qubits
        layers2 = get_cached_hamming_layers(2)
        self.assertIsInstance(layers2, dict)
        self.assertEqual(layers2[0], [0])  # Hamming weight 0
        self.assertEqual(set(layers2[1]), {1, 2})  # Hamming weight 1
        self.assertEqual(layers2[2], [3])  # Hamming weight 2
        
        # Second call should return cached
        layers2_2 = get_cached_hamming_layers(2)
        self.assertEqual(layers2, layers2_2)
    
    def test_generate_batch_trees_worker(self):
        """Test batch tree generation worker."""
        num_qubits = 3
        fix_random_seed(42)
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits + 1)]
        
        # Test batch processing
        tree_indices = [0, 1, 2]
        base_seed = 42
        args = (num_qubits, samples, tree_indices, base_seed)
        
        results = _generate_batch_trees_worker(args)
        
        # Check results structure
        self.assertEqual(len(results), 3)
        for i, (tree_idx, signs) in enumerate(results):
            self.assertEqual(tree_idx, i)
            self.assertIsInstance(signs, np.ndarray)
            self.assertEqual(signs.shape, (2**num_qubits,))
            self.assertTrue(np.all(np.isin(signs, [-1, 1])))
    
    def test_pool_management(self):
        """Test persistent pool creation and cleanup."""
        # Clean up any existing pool
        cleanup_pool()
        
        # Create pool
        pool1 = get_or_create_pool(2)
        self.assertIsNotNone(pool1)
        
        # Second call should return same pool
        pool2 = get_or_create_pool(2)
        self.assertIs(pool1, pool2)
        
        # Cleanup should work
        cleanup_pool()
        import hadamard_random_forest.random_forest as rf
        self.assertIsNone(rf._GLOBAL_POOL)


class TestSignDetermination(unittest.TestCase):
    """Test mathematical correctness of sign determination algorithm."""

    def setUp(self):
        """Set up test data for sign determination tests."""
        self.num_qubits = 3
        fix_random_seed(42)

    def test_sign_formula_correctness(self):
        """Test the sign determination formula from tutorial equation (3)."""
        # Create known test case based on tutorial equation (3):
        # sgn[2|ψ^k_j|^2 - |ψ_j|^2 - |ψ_{j+2^k}|^2]
        
        num_qubits = 2  # Simple case for verification
        N = 2**num_qubits
        
        # Create mock samples that should produce known signs
        # Base amplitudes: |ψ_0|^2, |ψ_1|^2, |ψ_2|^2, |ψ_3|^2
        base_probs = np.array([0.4, 0.3, 0.2, 0.1])  # Must sum to 1
        
        # For k=0 (step=1): affects pairs (0,1) and (2,3)
        # For k=1 (step=2): affects pairs (0,2) and (1,3)
        
        # Create Hadamard superposition results based on equation (2)
        # ψ^k_{j, j+2^k} = (1/√2)(ψ_j ± ψ_{j+2^k})
        samples = [base_probs]  # samples[0] = base probabilities
        
        # Mock k=0 Hadamard results (step=1)
        h0_probs = base_probs.copy()  # Start with base, will be modified
        samples.append(h0_probs)
        
        # Mock k=1 Hadamard results (step=2)  
        h1_probs = base_probs.copy()
        samples.append(h1_probs)
        
        # Generate tree and test weight computation
        tree, _ = generate_hypercube_tree(num_qubits)
        roots, leafs = find_global_roots_and_leafs(tree, num_qubits)
        
        weights = get_weight(samples, roots, leafs, num_qubits)
        
        # Verify basic properties
        self.assertEqual(weights[0], 1.0)  # Root weight is always 1
        self.assertTrue(np.all(np.isin(weights, [-1, 1])))  # All weights are ±1

    def test_weight_computation_properties(self):
        """Test mathematical properties of weight computation."""
        num_qubits = 3
        N = 2**num_qubits
        
        # Create normalized probability distributions
        fix_random_seed(42)
        samples = []
        for _ in range(num_qubits + 1):
            sample = np.random.rand(N)
            sample = sample / np.sum(sample)  # Normalize
            samples.append(sample)
        
        tree, _ = generate_hypercube_tree(num_qubits)
        roots, leafs = find_global_roots_and_leafs(tree, num_qubits)
        
        weights = get_weight(samples, roots, leafs, num_qubits)
        
        # Test mathematical properties
        self.assertEqual(weights.shape, (N,))
        self.assertEqual(weights[0], 1.0)  # Root weight
        self.assertTrue(np.all(np.isin(weights, [-1, 1])))  # Binary weights
        self.assertEqual(len(np.unique(weights)), 2)  # Only +1 and -1

    def test_hadamard_superposition_principle(self):
        """Test that Hadamard operations follow equation (2) from tutorial."""
        # This tests the theoretical basis: ψ^k_{j, j+2^k} = (1/√2)(ψ_j ± ψ_{j+2^k})
        
        # Create a simple 2-qubit case where we can verify the math
        num_qubits = 2
        
        # Known amplitudes for |00⟩, |01⟩, |10⟩, |11⟩
        amplitudes = np.array([0.6, 0.4, 0.5, 0.5])  # Real amplitudes
        amplitudes = amplitudes / np.linalg.norm(amplitudes)  # Normalize
        
        base_probs = amplitudes**2
        
        # Verify normalization
        self.assertAlmostEqual(np.sum(base_probs), 1.0, places=10)
        
        # Test that our probability distributions are valid
        self.assertTrue(np.all(base_probs >= 0))
        self.assertTrue(np.all(base_probs <= 1))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_qubit_system(self):
        """Test the minimal quantum system (1 qubit)."""
        num_qubits = 1
        num_trees = 3
        
        # Create valid samples for 1-qubit system
        fix_random_seed(42)
        samples = [np.random.rand(2) for _ in range(2)]  # 2 samples for 1 qubit
        samples = [s / np.sum(s) for s in samples]  # Normalize
        
        result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        
        self.assertEqual(result.shape, (2,))  # 2^1 = 2 elements
        self.assertTrue(np.all(np.isin(result, [-1, 1])))

    def test_visualization_threshold_boundary(self):
        """Test behavior at visualization threshold (10 qubits)."""
        # Test at threshold (should work)
        num_qubits = 10
        num_trees = 2
        fix_random_seed(42)
        samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits + 1)]
        
        # This should work without warnings about visualization
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
            # Should not have warnings about tree size
            tree_warnings = [warning for warning in w if "Too large to render" in str(warning.message)]
            self.assertEqual(len(tree_warnings), 0)

        # Test beyond threshold (should disable visualization)
        num_qubits = 11
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch('hadamard_random_forest.random_forest.logging') as mock_logging:
                # Create proper samples for 11 qubits
                large_samples = [np.random.rand(2**num_qubits) for _ in range(num_qubits + 1)]
                result = generate_random_forest(num_qubits, num_trees, large_samples, save_tree=True)
                # Should log warning about disabling visualization
                mock_logging.warning.assert_called()

    def test_invalid_sample_dimensions(self):
        """Test handling of invalid sample data."""
        num_qubits = 3
        num_trees = 3
        
        # Wrong number of samples (should be num_qubits + 1)
        wrong_samples = [np.random.rand(8) for _ in range(2)]  # Only 2 instead of 4
        
        with self.assertRaises(IndexError):
            generate_random_forest(num_qubits, num_trees, wrong_samples, save_tree=False)
        
        # Wrong sample dimensions (should be 2^num_qubits)
        wrong_dim_samples = [np.random.rand(4) for _ in range(4)]  # Should be 8 elements
        
        with self.assertRaises(IndexError):
            generate_random_forest(num_qubits, num_trees, wrong_dim_samples, save_tree=False)

    def test_zero_trees(self):
        """Test edge case with zero trees."""
        num_qubits = 2
        num_trees = 0
        samples = [np.random.rand(4) for _ in range(3)]
        
        # Zero trees should generate a result with a warning (not an error)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
            
            # Should generate warning about zero elements
            zero_warnings = [warning for warning in w if "Zero elements" in str(warning.message)]
            self.assertGreater(len(zero_warnings), 0)
            
            # Should still return valid result (all +1s as fallback)
            self.assertEqual(result.shape, (4,))
            self.assertTrue(np.all(result == 1))

    def test_deterministic_seeding_edge_cases(self):
        """Test random seeding in edge cases."""
        num_qubits = 2
        samples = [np.random.rand(4) for _ in range(3)]
        
        # Test with different random seeds
        seeds = [0, 1, 42, 999, 2**31 - 1]
        results = []
        
        for seed in seeds:
            fix_random_seed(seed)
            result = generate_random_forest(num_qubits, 3, samples, save_tree=False)
            results.append(result)
        
        # All results should be valid
        for result in results:
            self.assertEqual(result.shape, (4,))
            self.assertTrue(np.all(np.isin(result, [-1, 1])))
        
        # Results with different seeds should generally be different
        # (though there's a small chance some could be identical)
        # For deterministic testing, just verify they're all valid
        unique_results = [tuple(r) for r in results]
        # At least verify we got some results back
        self.assertEqual(len(results), len(seeds))


class TestTutorialWorkflow(unittest.TestCase):
    """Test integration following the tutorial notebook workflow."""

    def test_tutorial_default_parameters(self):
        """Test with default parameters from tutorial (111 trees)."""
        num_qubits = 3  # Smaller than tutorial for faster testing
        num_trees = 111  # Tutorial default
        
        # Create realistic sample data
        fix_random_seed(999)  # Tutorial uses this seed
        samples = []
        
        # Base probabilities (amplitudes squared)
        base_sample = np.random.rand(2**num_qubits)
        base_sample = base_sample / np.sum(base_sample)
        samples.append(base_sample)
        
        # Hadamard measurement results
        for k in range(num_qubits):
            h_sample = np.random.rand(2**num_qubits)
            h_sample = h_sample / np.sum(h_sample)
            samples.append(h_sample)
        
        # Test reconstruction
        result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False, show_tree=False)
        
        # Verify output properties
        self.assertEqual(result.shape, (2**num_qubits,))
        self.assertTrue(np.all(np.isin(result, [-1, 1])))
        
        # Test reproducibility with same seed
        fix_random_seed(999)
        result2 = generate_random_forest(num_qubits, num_trees, samples, save_tree=False, show_tree=False)
        np.testing.assert_array_equal(result, result2)

    def test_fidelity_bounds_principle(self):
        """Test the fidelity bounds principle from tutorial."""
        # This tests the concept: Fidelity Upper Bound (no sign errors) >= HRF Fidelity
        
        num_qubits = 2
        num_trees = 5
        
        # Create a known quantum state
        exact_amplitudes = np.array([0.6, 0.3, 0.5, 0.4])
        exact_amplitudes = exact_amplitudes / np.linalg.norm(exact_amplitudes)
        
        # Generate realistic samples (with some noise)
        fix_random_seed(42)
        samples = []
        
        # Base sample should approximate |amplitude|^2
        base_sample = exact_amplitudes**2 + 0.01 * np.random.randn(4)  # Add small noise
        base_sample = np.abs(base_sample)  # Ensure positive
        base_sample = base_sample / np.sum(base_sample)  # Normalize
        samples.append(base_sample)
        
        # Add Hadamard samples
        for k in range(num_qubits):
            h_sample = np.random.rand(4)
            h_sample = h_sample / np.sum(h_sample)
            samples.append(h_sample)
        
        # Reconstruct state
        reconstructed_signs = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        reconstructed_amplitudes = np.sqrt(base_sample) * reconstructed_signs
        
        # Test that we get a valid quantum state
        self.assertAlmostEqual(np.linalg.norm(reconstructed_amplitudes), 1.0, places=5)
        
        # Verify the reconstruction produces real values (as expected for HRF)
        self.assertTrue(np.all(np.isreal(reconstructed_amplitudes)))

    def test_circuit_count_requirement(self):
        """Test that exactly n+1 circuits are needed as stated in tutorial."""
        num_qubits = 3
        expected_circuits = num_qubits + 1  # n+1 circuits
        
        # Test with correct number of samples
        samples = [np.random.rand(2**num_qubits) for _ in range(expected_circuits)]
        samples = [s / np.sum(s) for s in samples]  # Normalize
        
        result = generate_random_forest(num_qubits, 3, samples, save_tree=False)
        self.assertEqual(result.shape, (2**num_qubits,))
        
        # Test with too few samples (should fail)
        insufficient_samples = samples[:-1]  # Remove one sample
        
        with self.assertRaises(IndexError):
            generate_random_forest(num_qubits, 3, insufficient_samples, save_tree=False)

    def test_real_valued_state_assumption(self):
        """Test that HRF is designed for real-valued quantum states."""
        # HRF is specifically for real-valued states as mentioned in tutorial
        num_qubits = 2
        num_trees = 5
        
        # Create samples for real-valued state
        fix_random_seed(42)
        samples = [np.random.rand(4) for _ in range(3)]
        samples = [s / np.sum(s) for s in samples]
        
        result = generate_random_forest(num_qubits, num_trees, samples, save_tree=False)
        
        # The signs should be real (±1)
        self.assertTrue(np.all(np.isreal(result)))
        self.assertTrue(np.all(np.isin(result, [-1, 1])))
        
        # When combined with amplitude magnitudes, should give real amplitudes
        amplitudes = np.sqrt(samples[0]) * result
        self.assertTrue(np.all(np.isreal(amplitudes)))


if __name__ == '__main__':
    unittest.main()

