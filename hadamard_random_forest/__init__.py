"""
hadamard_random_forest

Top‑level package for the Hadamard Random Forest (HRF) toolkit.
Provides functions for constructing and analyzing random spanning trees
on hypercube graphs, sampling circuits (both simulator and hardware),
and reconstructing real-valued quantum statevectors via majority‑voting.
"""
from __future__ import annotations

# Package Metadata
__title__ = "hadamard_random_forest"
__version__ = "0.1.0"
__license__ = "MIT"

# Core functionality from main module
from .main import (
    fix_random_seed,
    optimized_uniform_spanning_tree,
    generate_hypercube_tree,
    generate_random_forest,
    get_circuits,
    get_circuits_hardware,
    get_samples,
    get_samples_noisy,
    get_samples_hardware,
    get_statevector
)

# Utility functions
from .utils import (
    random_statevector,
    logarithmic_negativity,
    stabilizer_entropy,
    swap_test
)

__all__ = [
    # metadata
    "__title__", "__version__",  "__license__",
    # main API
    "fix_random_seed",
    "optimized_uniform_spanning_tree",
    "generate_hypercube_tree",
    "generate_random_forest",
    "get_circuits",
    "get_circuits_hardware",
    "get_samples",
    "get_samples_noisy",
    "get_samples_hardware",
    "get_statevector",
    # utilities
    "random_statevector",
    "logarithmic_negativity",
    "stabilizer_entropy",
    "swap_test"
]
