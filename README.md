## Quantum HRF Tomography
<img src="assets/logo-qht.png" alt="HRF Banner" width="300"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license) <a href="https://github.com/comp-physics/Quantum-HRF-Tomography/actions">
<img src="https://github.com/comp-physics/Quantum-HRF-Tomography/actions/workflows/ci.yml/badge.svg" /> </a> [![Coverage Status](https://coveralls.io/repos/github/comp-physics/Quantum-HRF-Tomography/badge.svg)](https://coveralls.io/github/comp-physics/Quantum-HRF-Tomography) [![arXiv](https://img.shields.io/badge/arXiv-2505.06455-b31b1b.svg)](https://arxiv.org/abs/2505.06455)

Efficient and Robust Reconstruction of Real-Valued Quantum States using **Hadamard Random Forests**

### Summary

⚡ **Fast**: Reduces required quantum circuits from **exponential** to **linear** in the number of qubits  
🎯 **Accurate**: Achieves 89% fidelity on 10-qubit real states using IBM quantum hardware  
🧠 **Smart**: Uses a random forest over hypercube graphs for efficient sign reconstruction  

### Introduction

This is the code that accompanies the following paper: [arxiv.org/abs/2505.06455](https://arxiv.org/abs/2505.06455)

<div align="center">
<img src="https://github.com/comp-physics/Quantum-HRF-Tomography/blob/master/assets/overview.png" height="300px"> 
</div>

Abstract:
> Quantum tomography is a crucial tool for characterizing quantum states and devices and estimating nonlinear properties of the systems. Performing full quantum state tomography on an N-qubit system requires an exponentially increasing overhead with O(3N) distinct Pauli measurement settings to resolve all complex phases and reconstruct the density matrix. However, many potential quantum computing applications, such as linear system solves, require only real-valued amplitudes. We introduce a readout method for real-valued quantum states that reduces the measurement settings required for state vector reconstruction to O(N); the post-processing cost remains exponential Ω(2N). This approach offers a substantial speedup over conventional tomography. We experimentally validate our method up to 10 qubits on the latest available IBM quantum processor and demonstrate that it accurately extracts key properties such as entanglement and magic. Our method also outperforms the standard SWAP test for state overlap estimation. This calculation resembles a numerical integration in certain cases and can be applied to extract nonlinear properties, which are important in application fields. We further implement the method to read out the solution from a quantum linear solver.

### Install 

We recommend cloning the repo. and installing locally:

```bash
git clone https://github.com/comp-physics/Quantum-HRF-Tomography
cd Quantum-HRF-Tomography
python -m venv qenv
source qenv/bin/activate
pip install -e .
pip install jupyter
```

To visualize the tree structure, one needs to install `Graphviz` to enforce the graph layout. For macOS,

```bash
brew install graphviz
pip install --global-option=build_ext \
            --global-option="-I$(brew --prefix graphviz)/include" \
            --global-option="-L$(brew --prefix graphviz)/lib" \
            pygraphviz
```

Please refer to [here](https://www.graphviz.org/download/) for more instructions. Then one can use 

```python
import hadamard_random_forest as hrf
hrf.get_statevector(num_qubits, num_trees, samples, save_tree=True, show_tree=True)
```

### Citation

```bibtex
@article{song2025reconstructing,
  author       = {Zhixin Song and Hang Ren and Melody Lee and Bryan Gard and Nicolas Renaud and Spencer H. Bryngelson},
  title        = {Hadamard {R}andom {F}orest: {R}econstructing real-valued quantum states with exponential reduction in measurement settings},
  year         = {2025},
  eprint       = {2505.06455},
  archivePrefix= {arXiv},
  primaryClass = {quant-ph}
}
```


### License

MIT
